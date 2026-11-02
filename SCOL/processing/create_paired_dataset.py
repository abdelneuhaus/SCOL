"""
Creation of paired (HIGH, LOW) patches to train CARE.
Detection part adapted from https://github.com/tmonseigne/palm-tracer
"""

import sys
import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Makes "processing" importable when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import processing.Parsing as Parsing
from processing.utils import load_dll, as_c_contig, max_allocation_bytes


DENSITY = 0.1  # max. expected localizations per pixel (sets the output buffer size)
C_IMG, C_TAB_DBL = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_double)
C_UINT, C_DBL = ctypes.c_uint64, ctypes.c_double


@lru_cache(maxsize=None)    # no need to reload the DLL at each call
def _get_dll() -> ctypes.CDLL:
    """
    Load the PALMTracer library (.dll on Windows, .dylib on macOS).

    Thanks to ``lru_cache``, the library is loaded at the first call only; the next calls
    return the same object.

    Returns:
        ctypes.CDLL: the loaded library.

    Raises:
        RuntimeError: if the library cannot be loaded.
    """
    dll = load_dll("CPU")   # .dll on Windows, .dylib on macOS
    if dll is None:
        raise RuntimeError("Failed to load PALMTracer DLL. Check path and permissions.")
    return dll


def detect_points_PALM(stack: np.ndarray, threshold: float, watershed: bool, fit: int,
                       fit_params: np.ndarray) -> pd.DataFrame:
    """
    Detect and fit molecules with PALMTracer on a (T, Y, X) stack or a (Y, X) image.

    The whole stack is sent to the PALMTracer library in one call, split into blocks
    of frames only if it does not fit in memory.

    Args:
        stack (np.ndarray): (T, Y, X) stack or (Y, X) image. Converted to uint16.
        threshold (float): detection threshold on the wavelet coefficients, in image
            intensity units (ADU).
        watershed (bool): if True, a watershed separates molecules that are close to each other.
        fit (int): PALMTracer fit mode. 0: no fit, 1: Gaussian (X, Y), 2: Gaussian (X, Y, sigma),
            3: Gaussian (X, Y, sigma_x, sigma_y), 4: Gaussian (X, Y, sigma_x, sigma_y, theta), 5: spline.
        fit_params (np.ndarray): fit parameters. For a Gaussian fit:
            [ROI size (pixels), initial sigma (pixels), 2 * initial sigma, theta (radians)].

    Returns:
        pd.DataFrame: one row per molecule, with columns ``Plane`` (frame index, starting at 0
        like numpy), ``X`` and ``Y`` (sub-pixel positions, in pixels).
    """
    dll = _get_dll()
    stk = as_c_contig(stack, np.dtype(np.uint16), writeable=False)
    if stk.ndim == 2:
        stk = stk[np.newaxis]                      # one image -> stack of one frame
    params = as_c_contig(fit_params, np.dtype(np.float64), writeable=False)
    n_frames, height, width = stk.shape

    # The stack is sent to the library by blocks of frames that fit in memory
    size_per_frame = int(height * width * DENSITY) * Parsing.N_COL_LOC
    block_len = max(1, int(max_allocation_bytes() // 8) // size_per_frame)

    tables = []
    for start in range(0, n_frames, block_len):
        block = np.ascontiguousarray(stk[start:start + block_len])
        k = block.shape[0]
        buffer = np.empty(size_per_frame * k, dtype=np.float64)
        count = dll.Localization(block.ctypes.data_as(C_IMG), buffer.ctypes.data_as(C_TAB_DBL),
                                 C_UINT(buffer.size), C_UINT(height), C_UINT(width), C_UINT(k),
                                 C_DBL(threshold), C_DBL(0 if watershed else 10), C_UINT(fit),
                                 params.ctypes.data_as(C_TAB_DBL))
        res = Parsing.parse_result(buffer[:count], "Localization")
        res["Plane"] += start - 1                  # PALMTracer counts frames from 1 in each block
        tables.append(res[["Plane", "X", "Y"]])

    locs = pd.concat(tables, ignore_index=True)
    locs["Plane"] = locs["Plane"].astype(int)
    return locs



def _check_frame_numbering(stack: np.ndarray, locs: pd.DataFrame, **detect_kwargs) -> None:
    """
    Check that the frame numbers returned by the whole-stack detection are correct.

    The frame with the most molecules is processed alone; the molecules found must be exactly
    those found for this frame when the whole stack was processed. A shift of one frame would
    otherwise pair the HIGH patch of a frame with the LOW patches of another one, without any error.

    Args:
        stack (np.ndarray): (T, Y, X) stack given to the detection.
        locs (pd.DataFrame): detections of the whole stack, as returned by ``detect_points_PALM``
            (columns ``Plane``, ``X``, ``Y``).
        **detect_kwargs: detection parameters (``threshold``, ``watershed``, ``fit``, ``fit_params``),
            the same as those used for ``locs``.

    Raises:
        RuntimeError: if the frame processed alone gives different molecules.
    """
    if locs.empty:
        return
    f = int(locs["Plane"].mode()[0])               # frame with the most molecules
    alone = detect_points_PALM(stack[f], **detect_kwargs)
    in_stack = locs[locs["Plane"] == f]
    same = len(alone) == len(in_stack) and np.allclose(np.sort(alone["X"]), np.sort(in_stack["X"]))
    if not same:
        raise RuntimeError(f"Frame numbering problem: frame {f} differs when processed alone.")



def _allowed_area(shape: tuple, size_ROI_crop: int, border_margin: int, margin_t: int) -> dict:
    """
    Compute where a patch center can be.

    A patch centered there stays inside the image (minus the border), and its temporal window
    stays inside the movie.

    Args:
        shape (tuple): (T, Y, X) shape of the stack.
        size_ROI_crop (int): patch size, in pixels.
        border_margin (int): pixels excluded at the image border.
        margin_t (int): number of frames needed before and after the central frame.

    Returns:
        dict: allowed intervals (bounds included) of the patch center:
        ``"f"`` for the frame, ``"x"`` and ``"y"`` for the position.
    """
    n_frames, H, W = shape
    half = size_ROI_crop // 2
    return {"f": (margin_t, n_frames - 1 - margin_t),
            "x": (border_margin + half, W - border_margin - half),
            "y": (border_margin + half, H - border_margin - half)}


def _usable_molecules(high: np.ndarray, mask: np.ndarray, area: dict, threshold: float,
                      size_ROI_fit: int) -> pd.DataFrame:
    """
    Detect the molecules on the HIGH stack and keep those that can be used as patch centers.

    Args:
        high (np.ndarray): (T, Y, X) HIGH stack.
        mask (np.ndarray): (Y, X) mask; patch centers must be on non-zero pixels.
        area (dict): allowed area, as returned by ``_allowed_area``.
        threshold (float): PALMTracer detection threshold.
        size_ROI_fit (int): size of the fitting ROI, in pixels.

    Returns:
        pd.DataFrame: usable molecules (columns ``Plane``, ``X``, ``Y``).
    """
    sigma = 1.0                                       # initial sigma of the Gaussian fit (pixels)
    detect_kwargs = dict(threshold=threshold, watershed=True, fit=2,
                         fit_params=np.array([size_ROI_fit, sigma, 2 * sigma, 0.0]))
    locs = detect_points_PALM(high, **detect_kwargs)
    _check_frame_numbering(high, locs, **detect_kwargs)
    inside = locs["Plane"].between(*area["f"]) & locs["X"].between(*area["x"]) & locs["Y"].between(*area["y"])
    locs = locs[inside]
    return locs[mask[locs["Y"].astype(int), locs["X"].astype(int)] > 0]


def _molecule_centers(locs: pd.DataFrame, n: int, area: dict, shift_max: int,
                      rng: np.random.Generator) -> list:
    """
    Draw n molecules at random and use them as patch centers.

    Each center is shifted at random by up to +/- shift_max pixels (so that the molecule is not
    always in the middle of the patch), then brought back inside the allowed area if needed.

    Args:
        locs (pd.DataFrame): usable molecules (columns ``Plane``, ``X``, ``Y``).
        n (int): number of centers to draw (at most ``len(locs)``).
        area (dict): allowed area, as returned by ``_allowed_area``.
        shift_max (int): maximum random shift, in pixels.
        rng (np.random.Generator): random generator.

    Returns:
        list: n tuples (frame, x, y).
    """
    chosen = locs.sample(n=n, random_state=rng)
    dx = rng.integers(-shift_max, shift_max + 1, n)
    dy = rng.integers(-shift_max, shift_max + 1, n)
    x = np.clip(np.rint(chosen["X"].to_numpy() + dx), *area["x"])
    y = np.clip(np.rint(chosen["Y"].to_numpy() + dy), *area["y"])
    return list(zip(chosen["Plane"], x, y))


def _background_centers(mask: np.ndarray, n: int, area: dict, rng: np.random.Generator) -> list:
    """
    Draw n random patch centers: random frame, random allowed position inside the mask.

    Args:
        mask (np.ndarray): (Y, X) mask; centers are drawn on non-zero pixels.
        n (int): number of centers to draw.
        area (dict): allowed area, as returned by ``_allowed_area``.
        rng (np.random.Generator): random generator.

    Returns:
        list: n tuples (frame, x, y).

    Raises:
        AssertionError: if no allowed position is inside the mask.
    """
    (x0, x1), (y0, y1) = area["x"], area["y"]
    ys, xs = np.nonzero(mask[y0:y1 + 1, x0:x1 + 1])
    assert len(xs) > 0, "No allowed patch center inside the mask."
    k = rng.integers(0, len(xs), n)
    frames = rng.integers(area["f"][0], area["f"][1] + 1, n)
    return list(zip(frames, xs[k] + x0, ys[k] + y0))


def _crop(high: np.ndarray, low: np.ndarray, f: int, x: int, y: int, size: int,
          t_high: int, t_low: int) -> tuple:
    """
    Cut the same patch in the HIGH and LOW stacks.

    Args:
        high (np.ndarray): (T, Y, X) HIGH stack.
        low (np.ndarray): (T, Y, X) LOW stack.
        f (int): central frame.
        x (int): column of the patch center.
        y (int): row of the patch center.
        size (int): patch size, in pixels.
        t_high (int): number of HIGH frames, centered on f (odd).
        t_low (int): number of LOW frames, centered on f (odd).

    Returns:
        tuple: HIGH patch (t_high, size, size) and LOW patch (t_low, size, size).
    """
    f, x, y, h = int(f), int(x), int(y), size // 2
    rows, cols = slice(y - h, y + h), slice(x - h, x + h)
    return (high[f - t_high // 2: f + t_high // 2 + 1, rows, cols],
            low[f - t_low // 2: f + t_low // 2 + 1, rows, cols])


def _save(stack_high: np.ndarray, stack_low: np.ndarray, output_dir: str, output_prefix: str) -> None:
    """
    Write the patches to output_dir/High/<prefix>_high_ROIs.tif and output_dir/Low/<prefix>_low_ROIs.tif.

    Integer data (camera images) are saved as uint16, other data as float32.

    Args:
        stack_high (np.ndarray): (N, t_high, P, P) HIGH patches.
        stack_low (np.ndarray): (N, t_low, P, P) LOW patches.
        output_dir (str): output folder; the High and Low sub-folders are created if needed.
        output_prefix (str): prefix of the two file names.
    """
    dtype = np.uint16 if stack_high.dtype == stack_low.dtype == np.uint16 else np.float32
    for sub, arr in (("High", stack_high), ("Low", stack_low)):
        folder = Path(output_dir) / sub
        folder.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(folder / f"{output_prefix}_{sub.lower()}_ROIs.tif", arr.astype(dtype), imagej=True)
    print(f"{len(stack_high)} patches saved in {Path(output_dir)}")




def build_paired_roi_stacks_batch(paths_high: list, paths_low: list, paths_mask: list = None, *,
                                  n_patches: int = 2000, t_window_high: int = 1, t_window_low: int = 3,
                                  threshold: float = 180.0, size_ROI_fit: int = 7, size_ROI_crop: int = 64,
                                  border_margin: int = 5, seed: int = None, output_dir: str = "Training",
                                  output_prefix: str = "batch", ratio_loc: float = 0.75) -> tuple:
    """
    Create paired (HIGH, LOW) patches to train CARE.

    The molecules are detected on the HIGH stacks. A fraction ``ratio_loc`` of the patches is
    centered (with a random shift of +/- size/4) on a molecule; the other patches are drawn at
    random positions inside the mask. The HIGH stacks must already be aligned on the LOW stacks,
    which stay raw.

    Args:
        paths_high (list): paths of the HIGH (target) stacks.
        paths_low (list): paths of the LOW (input) stacks, in the same order.
        paths_mask (list, optional): paths of 2D masks, one per stack. None: no mask.
        n_patches (int): total number of patches, split equally between the movies.
        t_window_high (int): number of HIGH frames per patch (odd). 1: N-to-1 model.
        t_window_low (int): number of LOW frames per patch (odd). 1: no temporal context.
        threshold (float): PALMTracer detection threshold, on the HIGH stacks.
        size_ROI_fit (int): size of the PALMTracer fitting ROI, in pixels.
        size_ROI_crop (int): patch size, in pixels.
        border_margin (int): pixels excluded at the image border.
        seed (int, optional): random seed; the same seed gives the same patches.
        output_dir (str): patches are written to output_dir/High and output_dir/Low.
        output_prefix (str): prefix of the two output files.
        ratio_loc (float): fraction of the patches centered on a molecule (between 0 and 1).

    Returns:
        tuple: HIGH patches (N, t_window_high, P, P) and LOW patches (N, t_window_low, P, P).
    """
    assert t_window_high % 2 == 1 and t_window_low % 2 == 1, "Temporal windows must be odd (1, 3, 5...)."
    assert len(paths_high) == len(paths_low), "paths_high and paths_low must have the same length."
    rng = np.random.default_rng(seed)
    margin_t = max(t_window_high, t_window_low) // 2      # frames needed before/after the central frame
    n = n_patches // len(paths_high)                      # patches per movie
    pairs = []

    for i in range(len(paths_high)):
        high, low = tifffile.imread(paths_high[i]), tifffile.imread(paths_low[i])
        assert high.shape == low.shape, f"High {high.shape} and low {low.shape} differ: pairs would be misaligned."
        mask = tifffile.imread(paths_mask[i]) if paths_mask else np.ones(high.shape[1:], dtype=bool)
        area = _allowed_area(high.shape, size_ROI_crop, border_margin, margin_t)

        print(f"Movie {i + 1}/{len(paths_high)}: searching for molecules...")
        locs = _usable_molecules(high, mask, area, threshold, size_ROI_fit)
        n_loc = min(round(n * ratio_loc), len(locs))
        n_bg = n - n_loc
        print(f"  {len(locs)} usable molecules -> {n_loc} patches on a molecule, {n_bg} background patches")

        centers = _molecule_centers(locs, n_loc, area, size_ROI_crop // 4, rng) + _background_centers(mask, n_bg, area, rng)
        pairs += [_crop(high, low, f, x, y, size_ROI_crop, t_window_high, t_window_low) for f, x, y in centers]

    rng.shuffle(pairs)                                    # same order for HIGH and LOW: pairs stay together
    stack_high = np.stack([h for h, l in pairs])
    stack_low = np.stack([l for h, l in pairs])
    _save(stack_high, stack_low, output_dir, output_prefix)
    return stack_high, stack_low




if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data" / "SIMULATION"

    params = dict(t_window_high=1, t_window_low=3, threshold=15.0, size_ROI_fit=8,
                  size_ROI_crop=64, border_margin=5, ratio_loc=0.75, seed=0)

    build_paired_roi_stacks_batch([DATA / "High.tif"], 
                                  [DATA / "Low.tif"], 
                                  n_patches=2000,
                                  output_dir=DATA / "Training", 
                                  output_prefix="train", **params)