import time
import numpy as np
import tifffile as tiff
from scipy.ndimage import map_coordinates


def read_coefficients_from_file(file_path: str):
    """
    Open and parse the coefficients used for the subpixel transformation.

    Args:
        file_path (str): Path to the coefficient file (PALMTracer 2CFit format:
            line 2 holds the x coefficients, line 3 the y coefficients).

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y coefficient arrays (10 values each)
            of the third-order polynomial.

    Raises:
        ValueError: If the file has fewer than 3 lines.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("File must have at least 3 lines.")
    coeff_x = np.array(list(map(float, lines[1].split())), dtype=np.float64)
    coeff_y = np.array(list(map(float, lines[2].split())), dtype=np.float64)
    return coeff_x, coeff_y


def polynomial(x: np.ndarray, y: np.ndarray, cfx: np.ndarray, cfy: np.ndarray):
    """
    Evaluate the third-order polynomial transformation on arrays of coordinates.

    Term order follows the PALMTracer 2CFit convention:
    x^3, y^3, x^2 y, x y^2, x^2, y^2, x y, x, y, 1.

    Args:
        x (np.ndarray): x coordinates (any shape).
        y (np.ndarray): y coordinates (same shape as ``x``).
        cfx (np.ndarray): 10 polynomial coefficients for the x output.
        cfy (np.ndarray): 10 polynomial coefficients for the y output.

    Returns:
        tuple[np.ndarray, np.ndarray]: Transformed (new_x, new_y) coordinates,
            same shape as the inputs.
    """
    x2, y2 = x * x, y * y
    x3, y3 = x2 * x, y2 * y
    terms = (x3, y3, x2 * y, x * y2, x2, y2, x * y, x, y, np.ones_like(x))
    new_x = sum(c * t for c, t in zip(cfx, terms))
    new_y = sum(c * t for c, t in zip(cfy, terms))
    return new_x, new_y


def build_sampling_grid(h: int, w: int, cfx: np.ndarray, cfy: np.ndarray):
    """
    Compute, for every pixel of the output grid, where to sample the source image.

    This is the backward-mapping grid: output pixel (x, y) takes the value of the
    source image interpolated at ``polynomial(x, y)``. The grid depends only on
    the image size and the coefficients, so it is computed once per stack.

    Note:
        The polynomial must map the output grid to the source image. To register
        the HIGH channel onto the LOW grid, use the LOW -> HIGH fit (the one
        previously used to forward-map the LOW channel).

    Args:
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        cfx (np.ndarray): x polynomial coefficients.
        cfy (np.ndarray): y polynomial coefficients.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - coords: array of shape (2, h*w) with source (row, col) positions,
              as expected by ``scipy.ndimage.map_coordinates``.
            - inside: boolean mask of shape (h, w), True where the sampling
              position falls inside the source image.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx, sy = polynomial(xx, yy, cfx, cfy)
    coords = np.stack([sy.ravel(), sx.ravel()])  # (row, col) order
    inside = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    return coords, inside


def warp_stack(stack: np.ndarray, coords: np.ndarray, order: int = 1, cval: float = 0.0):
    """
    Apply the subpixel transformation to a whole XYT stack by backward mapping.

    Each output pixel is interpolated once from the source frame, which avoids
    the hit-count normalisation, holes and Moiré patterns of forward mapping.

    Args:
        stack (np.ndarray): Input stack of shape (n, h, w).
        coords (np.ndarray): Sampling grid of shape (2, h*w) from
            :func:`build_sampling_grid`.
        order (int, optional): Interpolation order. 1 = bilinear (always
            non-negative, slight blur), 3 = cubic spline (sharper PSF, may
            overshoot slightly). Defaults to 1.
        cval (float, optional): Value assigned to pixels sampled outside the
            source image. Defaults to 0.0.

    Returns:
        np.ndarray: Warped stack of shape (n, h, w), dtype float32.
    """
    n, h, w = stack.shape
    out = np.empty((n, h, w), dtype=np.float32)
    for i in range(n):
        out[i] = map_coordinates(
            stack[i].astype(np.float32), coords,
            order=order, mode="constant", cval=cval, prefilter=(order > 1),
        ).reshape(h, w)
    return out


def subpixel_transformation(image_path: str, coeff_path: str, output_path: str,
                            order: int = 1, save_mask: bool = True):
    """
    Main function: load a stack, register it with the polynomial fit, save it.

    Out-of-field pixels are filled with the median of the first frame (camera
    background) rather than 0. A validity mask is optionally saved alongside
    the output so that LOW and HIGH can be cropped to the common valid field
    before patch extraction.

    Args:
        image_path (str): Path to the TIFF stack to shift.
        coeff_path (str): Path to the coefficient file.
        output_path (str): Path of the shifted TIFF stack to write.
        order (int, optional): Interpolation order passed to
            :func:`warp_stack`. Defaults to 1.
        save_mask (bool, optional): If True, write ``<output>_validmask.tif``
            (255 inside the valid field, 0 outside). Defaults to True.
    """
    t0 = time.time()
    print("Data Loading...")
    stack = tiff.imread(image_path)
    if stack.ndim == 2:
        stack = stack[None]
    cfx, cfy = read_coefficients_from_file(coeff_path)
    n, h, w = stack.shape
    print(f"{n} images to shift…")

    coords, inside = build_sampling_grid(h, w, cfx, cfy)
    cval = float(np.median(stack[0]))

    print("Processing...")
    warped = warp_stack(stack, coords, order=order, cval=cval)
    warped = np.clip(np.round(warped), 0, 65535).astype(np.uint16)

    tiff.imwrite(output_path, warped)
    if save_mask:
        tiff.imwrite(output_path.replace(".tif", "_validmask.tif"),
                     inside.astype(np.uint8) * 255)
    print(f"Image saved. Valid field: {inside.mean() * 100:.1f} % of pixels.")
    print(f"Finished in {time.time() - t0:.2f} second(s).")


if __name__ == "__main__":
    image_path  = "high.tif"
    coeff_path  = "data/NUP/SplitViewBeads.PT/SplitViewBeads_2CFit.txt"
    output_path = "high_shifted.tif"
    subpixel_transformation(image_path, coeff_path, output_path, order=1)