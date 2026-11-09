"""
Paired (HIGH, LOW) patches plus a localization mask, to test the L5 loss.

Experimental: same patches as build_paired_roi_stacks_batch (create_paired_dataset.py, not modified), 
plus a third stack: for each patch, a mask equal to 1 in a square of side mask_size around each molecule detected by PALMTracer 
on the HIGH frame of the patch, 0 elsewhere. The mask is only read by the loss during training; the network never sees it.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

# Makes "processing" importable when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.create_paired_dataset import (detect_points_PALM, _check_frame_numbering, _allowed_area,
                                              _usable_molecules, _molecule_centers, _background_centers,
                                              _crop, _save)


# PALMTracer positions: pixel i covers [i, i + 1), so its center is at i + 0.5 (same convention as
# _usable_molecules, which reads mask[int(Y), int(X)]). If the squares of the visual check look
# shifted by half a pixel, set this to 0.0.
PIXEL_CENTER = 0.0


def _all_detections(high: np.ndarray, threshold: float, size_ROI_fit: int) -> dict:
    """
    Detect all the molecules of the HIGH stack, frame by frame (no border or mask filtering).

    Args:
        high (np.ndarray): (T, Y, X) HIGH stack.
        threshold (float): PALMTracer detection threshold.
        size_ROI_fit (int): size of the fitting ROI, in pixels.

    Returns:
        dict: frame index -> (N, 2) array of (X, Y) positions, in pixels.
    """
    sigma = 1.0                                       # initial sigma of the Gaussian fit (pixels), as in _usable_molecules
    detect_kwargs = dict(threshold=threshold, watershed=True, fit=2,
                         fit_params=np.array([size_ROI_fit, sigma, 2 * sigma, 0.0]))
    locs = detect_points_PALM(high, **detect_kwargs)
    _check_frame_numbering(high, locs, **detect_kwargs)
    return {int(f): g[["X", "Y"]].to_numpy() for f, g in locs.groupby("Plane")}


def _draw_mask(positions: np.ndarray, x: int, y: int, size: int, mask_size: int) -> np.ndarray:
    """
    Mask of one patch: 1 in a mask_size x mask_size square around each detection, 0 elsewhere.

    The patch covers rows y - size/2 .. y + size/2 - 1 and columns x - size/2 .. x + size/2 - 1
    of the image (same as ``_crop``). Detections near the patch border give a partial square.

    Args:
        positions (np.ndarray): (N, 2) (X, Y) positions of the detections of the patch frame, in image pixels.
        x (int): column of the patch center.
        y (int): row of the patch center.
        size (int): patch size, in pixels.
        mask_size (int): side of the square around each detection, in pixels (6, 7, 8...).

    Returns:
        np.ndarray: (size, size) uint8 mask.
    """
    h = size // 2
    mask = np.zeros((size, size), dtype=np.uint8)
    centers = np.arange(size) + PIXEL_CENTER          # pixel centers, in patch coordinates
    for px, py in positions:
        u, v = px - (x - h), py - (y - h)             # detection in patch coordinates
        cols = np.abs(centers - u) < mask_size / 2
        rows = np.abs(centers - v) < mask_size / 2
        mask[np.ix_(rows, cols)] = 1
    return mask


def build_paired_roi_stacks_with_mask(paths_high: list, paths_low: list, paths_mask: list = None, *,
                                      n_patches: int = 2000, t_window_high: int = 1, t_window_low: int = 3,
                                      threshold: float = 180.0, size_ROI_fit: int = 7, size_ROI_crop: int = 64,
                                      border_margin: int = 5, seed: int = None, output_dir: str = "Training_L5",
                                      output_prefix: str = "patches", ratio_loc: float = 0.75,
                                      mask_size: int = 7, threshold_mask: float = None,
                                      ghost_sites: bool = True, ghost_min_dist: float = 3.0) -> tuple:
    """
    Create paired (HIGH, LOW) patches and their localization masks, to train CARE with the L5 loss.

    Same patches as ``build_paired_roi_stacks_batch``. In addition, all the molecules of the HIGH
    stack are detected (with ``threshold_mask``, lower than ``threshold`` if needed, so that faint
    molecules also get a square), and a mask is drawn for each patch on its central HIGH frame.

    Args:
        paths_high (list): paths of the HIGH (target) stacks.
        paths_low (list): paths of the LOW (input) stacks, in the same order.
        paths_mask (list, optional): paths of 2D masks restricting the patch centers. None: no mask.
            (Not the localization mask.)
        n_patches (int): total number of patches, split equally between the movies.
        t_window_high (int): number of HIGH frames per patch. Must be 1 (one mask per patch).
        t_window_low (int): number of LOW frames per patch (odd).
        threshold (float): PALMTracer detection threshold for the patch centers, on the HIGH stacks.
        size_ROI_fit (int): size of the PALMTracer fitting ROI, in pixels.
        size_ROI_crop (int): patch size, in pixels.
        border_margin (int): pixels excluded at the image border.
        seed (int, optional): random seed; the same seed gives the same patches.
        output_dir (str): patches are written to output_dir/High, output_dir/Low and output_dir/Mask.
        output_prefix (str): prefix of the three output files.
        ratio_loc (float): fraction of the patches centered on a molecule (between 0 and 1).
        mask_size (int): side of the square drawn around each detection, in pixels.
        threshold_mask (float, optional): PALMTracer detection threshold for the mask. None: same as threshold.
        ghost_sites (bool): also mark the ghost sites with 2: squares around the molecules of frames t - 1
            and t + 1, where there is no molecule square at t.
        ghost_min_dist (float): a molecule of t - 1 or t + 1 closer than this (pixels) to a molecule of t is
            the same molecule, still present: no ghost square.

    Returns:
        tuple: HIGH patches (N, 1, P, P), LOW patches (N, t_window_low, P, P) and masks (N, 1, P, P).
    """
    assert t_window_high == 1, "The localization mask needs one HIGH frame per patch (t_window_high = 1)."
    assert t_window_low % 2 == 1, "t_window_low must be odd (1, 3, 5...)."
    assert len(paths_high) == len(paths_low), "paths_high and paths_low must have the same length."
    threshold_mask = threshold if threshold_mask is None else threshold_mask
    rng = np.random.default_rng(seed)
    margin_t = t_window_low // 2
    n = n_patches // len(paths_high)
    triplets = []

    for i in range(len(paths_high)):
        high, low = tifffile.imread(paths_high[i]), tifffile.imread(paths_low[i])
        assert high.shape == low.shape, f"High {high.shape} and low {low.shape} differ: pairs would be misaligned."
        mask_centers = tifffile.imread(paths_mask[i]) if paths_mask else np.ones(high.shape[1:], dtype=bool)
        area = _allowed_area(high.shape, size_ROI_crop, border_margin, margin_t)

        print(f"Movie {i + 1}/{len(paths_high)}: searching for molecules...")
        locs = _usable_molecules(high, mask_centers, area, threshold, size_ROI_fit)
        detections = _all_detections(high, threshold_mask, size_ROI_fit)
        n_det = sum(len(p) for p in detections.values())
        n_loc = min(round(n * ratio_loc), len(locs))
        n_bg = n - n_loc
        print(f"  {len(locs)} usable molecules -> {n_loc} patches on a molecule, {n_bg} background patches")
        print(f"  {n_det} detections for the mask (threshold {threshold_mask})")

        centers = _molecule_centers(locs, n_loc, area, size_ROI_crop // 4, rng) + _background_centers(mask_centers, n_bg, area, rng)
        empty = np.empty((0, 2))
        for f, x, y in centers:
            h, l = _crop(high, low, f, x, y, size_ROI_crop, t_window_high, t_window_low)
            m = _draw_mask(detections.get(int(f), empty), int(x), int(y), size_ROI_crop, mask_size)
            if ghost_sites:
                # 2 = ghost site: a molecule at t - 1 or t + 1 with no molecule at t closer than ghost_min_dist
                # (the squares of t stay 1)
                now = detections.get(int(f), empty)
                near = np.concatenate([detections.get(int(f) - 1, empty), detections.get(int(f) + 1, empty)])
                if len(now) and len(near):
                    dist = np.linalg.norm(near[:, None, :] - now[None, :, :], axis=2).min(axis=1)
                    near = near[dist >= ghost_min_dist]
                g = _draw_mask(near, int(x), int(y), size_ROI_crop, mask_size)
                m[(g == 1) & (m == 0)] = 2
            triplets.append((h, l, m[np.newaxis]))

    rng.shuffle(triplets)                                 # same order for HIGH, LOW and mask
    stack_high = np.stack([t[0] for t in triplets])
    stack_low = np.stack([t[1] for t in triplets])
    stack_mask = np.stack([t[2] for t in triplets])
    _save(stack_high, stack_low, output_dir, output_prefix)
    folder = Path(output_dir) / "Mask"
    folder.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(folder / f"{output_prefix}_mask_ROIs.tif", stack_mask, imagej=True)

    info = {"date": datetime.now().isoformat(timespec="seconds"),
            "paths_high": [str(p) for p in paths_high], "paths_low": [str(p) for p in paths_low],
            "paths_mask": [str(p) for p in paths_mask] if paths_mask else None,
            "n_patches": len(stack_high), "t_window_high": t_window_high, "t_window_low": t_window_low,
            "threshold": threshold, "size_ROI_fit": size_ROI_fit, "size_ROI_crop": size_ROI_crop,
            "border_margin": border_margin, "ratio_loc": ratio_loc, "seed": seed,
            "mask_size": mask_size, "threshold_mask": threshold_mask,
            "ghost_sites": ghost_sites, "ghost_min_dist": ghost_min_dist,
            "mask_fraction": float((stack_mask == 1).mean()), "ghost_fraction": float((stack_mask == 2).mean()),
            "pixel_center": PIXEL_CENTER}
    with open(Path(output_dir) / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Masks: molecules {(stack_mask == 1).mean() * 100:.1f} %, ghost sites {(stack_mask == 2).mean() * 100:.1f} % "
          f"of the pixels, saved in {folder}")
    return stack_high, stack_low, stack_mask