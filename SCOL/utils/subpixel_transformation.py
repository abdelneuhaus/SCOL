import numpy as np
import tifffile as tiff
import time
from numba import njit, prange


def read_coefficients_from_file(file_path: str):
    """
    Open and parse coefficients used for subpixel transformation calculation.

    Args:
        path_file (str): Path to coefficient file.

    Returns:
        coeff_x, coeff_y (np.array): Arrays of x and y coefficients for polynomial funtion.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("File must have at least 3 lines.")
    coeff_x = np.array(list(map(float, lines[1].split())), dtype=np.float64)
    coeff_y = np.array(list(map(float, lines[2].split())), dtype=np.float64)
    return coeff_x, coeff_y



@njit(fastmath=True)
def calculate_new_coordinates(x, y, cfx, cfy):
    """
    Compute third order polynomial for x and y coordinates using coefficients. Can be done on int or float.

    Note:
        This function is decorated with @njit (Numba). First function call will be slower due to JIT compilation.

    Args:
        x (float | int): x coordinate value.
        y (float | int): y coordinate value.
        cfx (np.array): x coefficient file.
        cfy (np.array): y coefficient file.

    Returns:
        new_x, new_y (tuple[float, float]): New (x,y) coordinates with transformation applied.
    """
    x2, y2 = x * x, y * y
    x3, y3 = x2 * x, y2 * y

    new_x = (
        cfx[0] * x3 + cfx[1] * y3 + cfx[2] * x2 * y + cfx[3] * x * y2 +
        cfx[4] * x2 + cfx[5] * y2 + cfx[6] * x * y + cfx[7] * x +
        cfx[8] * y + cfx[9]
    )
    new_y = (
        cfy[0] * x3 + cfy[1] * y3 + cfy[2] * x2 * y + cfy[3] * x * y2 +
        cfy[4] * x2 + cfy[5] * y2 + cfy[6] * x * y + cfy[7] * x +
        cfy[8] * y + cfy[9]
    )
    return new_x, new_y



@njit(fastmath=True)  # dont put parallel true to not overwrite part of image
def shift_image_with_interpolation(image, cfx, cfy):
    """
    Apply subpixel transformation to one image by forward mapping. 

    Note:
        This function is decorated with @njit (Numba). First function call will be slower due to JIT compilation.
    
    Args:
        image (np.array): current frame
        cfx (np.array): x coefficient file
        cfy (np.array): y coefficient file

    Returns:
        out(np.array): shifted frame
    """
    h, w = image.shape
    out   = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    for y in prange(h):
        for x in range(w):
            nx, ny = calculate_new_coordinates(x, y, cfx, cfy)

            x0 = int(nx)
            y0 = int(ny)
            dx = nx - x0
            dy = ny - y0

            w00 = (1.0 - dx) * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w10 = dx * (1.0 - dy)
            w11 = dx * dy

            if 0 <= x0 < w and 0 <= y0 < h:
                out[y0, x0] += image[y, x] * w00
                count[y0, x0] += w00
            if 0 <= x0 < w and 0 <= y0 + 1 < h:
                out[y0 + 1, x0] += image[y, x] * w01
                count[y0 + 1, x0] += w01
            if 0 <= x0 + 1 < w and 0 <= y0 < h:
                out[y0, x0 + 1] += image[y, x] * w10
                count[y0, x0 + 1] += w10
            if 0 <= x0 + 1 < w and 0 <= y0 + 1 < h:
                out[y0 + 1, x0 + 1] += image[y, x] * w11
                count[y0 + 1, x0 + 1] += w11

    # normalize by number of hits in a pixel
    for y in prange(h):
        for x in range(w):
            if count[y, x] == 0.0:
                count[y, x] = 1.0
            out[y, x] /= count[y, x]

    return out



@njit(parallel=True, fastmath=True)
def process_stack(image_stack, cfx, cfy):
    """
    Perform subpixel alignment on the whole XYT stack. 

    Note:
        This function is decorated with @njit (Numba). First function call will be slower due to JIT compilation.
    
    Args:
        image_stack (np.array): whole stack to align
        cfx (np.array): x coefficient file
        cfy (np.array): y coefficient file

    Returns:
        out(np.array): shifted stack
    """
    n, h, w = image_stack.shape
    out = np.empty((n, h, w), dtype=np.float64)

    for i in prange(n):
        out[i] = shift_image_with_interpolation(image_stack[i], cfx, cfy)

    return out



def main(image_path, coeff_path, output_path):
    """
    Main function.
    
    Args:
        image_path (str): path to image to shift
        coeff_path (str): path to coefficients file
        output_path (str): path to save shifted image
    """

    t0 = time.time()
    print("Data Loading...")
    image_stack = tiff.imread(image_path)
    coeff_x, coeff_y = read_coefficients_from_file(coeff_path)

    print(f"{image_stack.shape[0]} images to shift…")

    # JIT compilation
    _ = process_stack(image_stack[:1], coeff_x, coeff_y)

    # processing
    print("Processing...")
    shifted_stack = process_stack(image_stack, coeff_x, coeff_y)
    
    # saving
    tiff.imwrite(output_path, shifted_stack.astype(np.uint16))
    print("Image saved.")
    print(f"Finished in {time.time() - t0:.2f} second(s).")




if __name__ == "__main__":
    image_path  = "//filer3/TEAM_M/everyone/_Transferts/Transfert Neuhaus/2D_NUP_DATASET/5mW/LOW/U2OS-NUP96_R4-100pM_015_low.tif"
    coeff_path  = "data/NUP/SplitViewBeads.PT/SplitViewBeads_2CFit.txt"
    output_path = "//filer3/TEAM_M/everyone/_Transferts/Transfert Neuhaus/2D_NUP_DATASET/5mW/LOW_SHIFTED/U2OS-NUP96_R4-100pM_015_low_shifted.tif"
    main(image_path, coeff_path, output_path)
