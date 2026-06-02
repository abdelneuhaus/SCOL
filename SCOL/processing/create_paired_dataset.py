"""
Adapted from https://github.com/tmonseigne/palm-tracer
Extraction of localization pipeline and adapted to standalone use.
"""

import random
import ctypes
import tifffile

import numpy as np
import time as time
import pandas as pd
import processing.Parsing as Parsing

from typing import Sequence, Tuple, Optional
from processing.utils import load_dll, as_c_contig, max_allocation_bytes


DENSITY = 0.1
C_IMG, C_TAB_DBL, C_TAB_UINT = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64)
C_UINT, C_BOOL, C_DBL = ctypes.c_uint64, ctypes.c_bool, ctypes.c_double



def detect_points_PALM(stack: np.ndarray, threshold: float, watershed: bool, fit: int, fit_params: np.ndarray,
					 planes: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Call function to perform advanced Gaussian fitting, part of PALMTracer.

    Args:
        frame (np.ndarray): SMLM image on which to perform Gaussian fitting
        dll (ctypes.CDLL): instance of the DLL loaded
        buff_ptr (ctypes.POINTER(ctypes.c_double)): pointer to output array (must be float64 or double)
        potential_points (int): maximal buffer size (number max of localization per plane)
        H (int): height of the image
        W (int): width of the image
        threhsold (float): threshold value to select molecules
        size_ROI_fit (int): size of the ROI to select molecules
    
    Returns:
        locs (List[Tuple[float, float]]): list of localized molecules XY coordinates
    """
    dll_type = "CPU"
    dll = load_dll(dll_type)
    
    if dll is None:
        raise RuntimeError("Failed to load PALMTracer DLL. Check path and permissions.")

    stk = as_c_contig(stack, np.dtype(np.uint16), writeable=False)
    params = as_c_contig(fit_params, np.dtype(np.float64), writeable=False)
    height, width = stk.shape[-2:]
	
    n_planes = 1 if stk.ndim == 2 else stk.shape[0]
    if planes is None: planes = list(range(n_planes))
    else: planes = [p for p in planes if 0 <= p < n_planes]
    n_planes = len(planes)
    stk = stk[np.newaxis, :, :] if stk.ndim == 2 else stk[planes[0]:planes[0] + n_planes]

    max_points = int(max_allocation_bytes() // 8)  # .				Nombre de points maximum allouable en une fois
    plane_points = int(height * width * DENSITY) * Parsing.N_COL_LOC  # Taille théorique max pour un seul plan (N points max * N Col localisation)
    if max_points < plane_points: return pd.DataFrame()  # . 			pragma: no cover — Cas extrême un seul plan est gargantuesque.
    n_plane_max = int(min(max_points // plane_points, n_planes))  # .	Nombre de plans qui tiennent dans max_allocation

    dfs: list[pd.DataFrame] = []
    i = 0
    while i < len(planes):
        k = min(n_plane_max, n_planes - i)  # .						Taille réelle du bloc, soit le max, soit "ce qui reste".
        stk_block = stk[i:i + k]  # .								Indices relatifs (0..n_planes-1)
        n_block = plane_points * k  # . 							Nombre de points pour ce bloc
        locs = np.empty((n_block,), dtype=np.float64, order="C")  # Création de la sortie

        count = dll.Localization(stk_block.ctypes.data_as(C_IMG), locs.ctypes.data_as(C_TAB_DBL), C_UINT(n_block), C_UINT(height), C_UINT(width),
                                    C_UINT(k), C_DBL(threshold), C_DBL(0 if watershed else 10), C_UINT(fit), params.ctypes.data_as(C_TAB_DBL))

        res = Parsing.parse_result(locs[:count], "Localization")
        if "Plane" in res.columns: res["Plane"] += planes[0] + i  # . En cas de filtre des plans, on incrémente par i + premier plan.
        dfs.append(res)
        i += k

    res = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if not res.empty:
        res.reset_index(drop=True, inplace=True)
        res["Id"] = res.index + 1  # .					  1-based comme attendu
        if fit == 4:  # Fit Gaussien avec Theta
            mask = res["Integrated Intensity"] > 0
            res.loc[mask, "Theta"] = Parsing.manage_theta(res.loc[mask, "Theta"])  # Clean Theta and show stats

    if res.empty:
        return []
        
    return list(zip(res['X'], res['Y']))




def build_paired_roi_stacks_batch(
    paths_high: Sequence[str],
    paths_low: Sequence[str],
    paths_mask: Optional[Sequence[str]] = None, 
    *,
    n_patches: Optional[int] = None,
    t_window_high: int = 5,
    t_window_low: int = 5,
    threshold: float = 180.0,
    size_ROI_fit: int = 7,
    size_ROI_crop: int = 8,
    border_margin: int = 5,
    seed: Optional[int] = None,
    output_prefix: str = "batch",
    ratio_loc: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract paired ROIs from high and low SNR stacks.

    Args:
        paths_high (Sequence[str]): List of file paths to the high SNR (target) TIFF stacks.
        paths_low (Sequence[str]): List of file paths to the low SNR (input) TIFF stacks.
        paths_mask (Optional[Sequence[str]]): List of paths to binary mask TIFFs.
        t_window (int): Temporal window size for the low SNR stacks. Must be an odd integer.
        threshold (float): Intensity threshold for the PALM point detection algorithm. Defaults to 180.0.
        size_ROI_fit (int): Diameter of the neighborhood (in pixels) used by the DLL for sub-pixel localization/fitting
        size_ROI_crop (int): The spatial dimensions (width and height) of the final extracted ROI patches.
        border_margin (int): Safety margin in pixels from the image edges to avoid out-of-bounds cropping
        dll_path (str): File path to the compiled C++ DLL (CPU_PALM) used for point detection.
        seed (Optional[int]): Random seed for reproducibility of the ROI jittering and background sampling
        output_prefix (str): Prefix used for the two saved TIFF files containing the aggregated ROIs
        ratio_loc (float): Proportion of patches with at least one detected molecule. The remaining patches are background from the image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - stacks_high: A 4D np array of shape (N, 1, size_ROI_crop, size_ROI_crop).
            - stacks_low: A 4D np array of shape (N, t_window, size_ROI_crop, size_ROI_crop).
    """
 
    assert t_window_high % 2 == 1, "t_window_high must be odd (eg: 1, 3, 5, 7)"
    assert t_window_low % 2 == 1, "t_window_low must be odd (eg: 1, 3, 5, 7)"
    assert 0 <= ratio_loc <= 1, "ratio_loc must be between 0 and 1"
    assert len(paths_high) == len(paths_low), "paths_high and paths_low must have the same length."

    margin_t_high = t_window_high // 2 
    margin_t_low = t_window_low // 2 

    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    rois_high, rois_low = [], []
    n_patches_per_stack = n_patches // len(paths_high) if n_patches is not None else None

    for i in range(len(paths_high)):
        path_A = paths_high[i]
        path_B = paths_low[i]
        stack_A = tifffile.imread(path_A)
        stack_B = tifffile.imread(path_B)
        n_frames, H, W = stack_A.shape

        if paths_mask is not None and paths_mask[i] is not None:
            mask = tifffile.imread(paths_mask[i])
        else:
            mask = np.ones((H, W), dtype=np.uint8)

        min_center_x = border_margin + size_ROI_crop // 2
        max_center_x = W - border_margin - size_ROI_crop // 2
        min_center_y = border_margin + size_ROI_crop // 2
        max_center_y = H - border_margin - size_ROI_crop // 2

        potential_y, potential_x = np.where(mask > 0)
        valid_idx = (potential_x >= min_center_x) & (potential_x <= max_center_x) & \
                    (potential_y >= min_center_y) & (potential_y <= max_center_y)
        
        valid_x = potential_x[valid_idx]
        valid_y = potential_y[valid_idx]

        all_loc_candidates = []

        # localization process
        sigma, theta = 1.0, 0.0
        fit_param = np.array([size_ROI_fit, sigma, 2 * sigma, theta], dtype=np.float64)
        print(f"Stack {i+1}/{len(paths_high)}: Searching for molecules...")
        for f in range(margin_t_low, n_frames - margin_t_low):
            frameA_center = stack_A[f] 
            locs = detect_points_PALM(frameA_center, threshold, True, 2, fit_param, list(range(0,10)))
            for cx, cy in locs:
                if (min_center_x <= cx <= max_center_x and min_center_y <= cy <= max_center_y):
                    if mask[int(cy), int(cx)] > 0:
                        all_loc_candidates.append((f, cx, cy))

        total_locs_found = len(all_loc_candidates)
        print(f"Number of potential molecules detected to crop around : {total_locs_found}")

        # % calculation
        if n_patches_per_stack is not None:
            target_loc = int(n_patches_per_stack * ratio_loc)
            target_bg = n_patches_per_stack - target_loc
        else:
            target_loc = total_locs_found
            target_bg = int(target_loc * (1 - ratio_loc) / max(ratio_loc, 1e-6)) if ratio_loc < 1 else 0

        if target_loc > total_locs_found:
            print(f"Warning: {target_loc} patches requested, but only {total_locs_found} found. Change ratio.")
            target_loc = total_locs_found
            if ratio_loc > 0:
                target_bg = int(target_loc * (1 - ratio_loc) / ratio_loc)

        print(f"OTarget : {target_loc} patchs with molecules, {target_bg} patchs with backround.")

        # random selection
        rng.shuffle(all_loc_candidates)
        selected_locs = all_loc_candidates[:target_loc]

        selected_bgs = []
        if len(valid_x) > 0 and target_bg > 0:
            for _ in range(target_bg):
                f_bg = rng.integers(margin_t_low, n_frames - margin_t_low)
                idx = rng.integers(0, len(valid_x))
                selected_bgs.append((f_bg, valid_x[idx], valid_y[idx]))

        # extraction
        crops_to_extract = [('loc', f, cx, cy) for f, cx, cy in selected_locs] + \
                           [('bg', f, cx, cy) for f, cx, cy in selected_bgs]
        rng.shuffle(crops_to_extract)

        for c_type, f, cx, cy in crops_to_extract:
            if c_type == 'loc':
                for _try in range(10):
                    dx = rng.integers(-size_ROI_crop // 4, size_ROI_crop // 4 + 1)
                    dy = rng.integers(-size_ROI_crop // 4, size_ROI_crop // 4 + 1)
                    x_c, y_c = int(round(cx + dx)), int(round(cy + dy))

                    x_min, x_max = x_c - size_ROI_crop // 2, x_c + size_ROI_crop // 2
                    y_min, y_max = y_c - size_ROI_crop // 2, y_c + size_ROI_crop // 2

                    if (border_margin <= x_min and border_margin <= y_min and 
                        x_max <= (W - border_margin) and y_max <= (H - border_margin)):
                        break
                else:
                    x_min, x_max = int(cx) - size_ROI_crop // 2, int(cx) + size_ROI_crop // 2
                    y_min, y_max = int(cy) - size_ROI_crop // 2, int(cy) + size_ROI_crop // 2
                    if (x_min < border_margin or y_min < border_margin or 
                        x_max > (W - border_margin) or y_max > (H - border_margin)):
                        continue
            else:
                x_min, x_max = int(cx) - size_ROI_crop // 2, int(cx) + size_ROI_crop // 2
                y_min, y_max = int(cy) - size_ROI_crop // 2, int(cy) + size_ROI_crop // 2

            rA = stack_A[f - margin_t_high : f + margin_t_high + 1, y_min:y_max, x_min:x_max]
            rB = stack_B[f - margin_t_low : f + margin_t_low + 1, y_min:y_max, x_min:x_max]
            rois_high.append(rA)
            rois_low.append(rB)

    if not rois_high:
        raise RuntimeError("No valid ROI found.")

    stack_high = np.stack(rois_high) 
    stack_low  = np.stack(rois_low)  

    perm = rng.permutation(stack_high.shape[0])
    stack_high = stack_high[perm]
    stack_low  = stack_low[perm]

    outA = f"{output_prefix}_high_ROIs.tif"
    outB = f"{output_prefix}_low_ROIs.tif"
    tifffile.imwrite(outA, stack_high.astype(np.float32), imagej=True)
    tifffile.imwrite(outB, stack_low.astype(np.float32), imagej=True)

    print(f"{stack_high.shape[0]} ROIs saved in {outA} and {outB}")
    return stack_high, stack_low