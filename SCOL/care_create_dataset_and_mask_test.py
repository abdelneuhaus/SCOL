import ctypes
import tifffile
import numpy as np
from typing import Sequence, Tuple, List, Optional
import random
import time as time


def _detect_points_PALM(frame: np.ndarray,
                        dll,
                        buff_ptr,
                        potential_points: int,
                        H: int, W: int,
                        threshold: float,
                        size_ROI_fit: int) -> List[Tuple[float, float]]:
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

    c_ushort_p = ctypes.POINTER(ctypes.c_ushort)
    img_ptr = frame.ctypes.data_as(c_ushort_p)

    waveletNo = ctypes.c_uint(1)
    thresholdVal = ctypes.c_double(threshold)
    watershedRatio = ctypes.c_double(0)
    volMin = ctypes.c_double(4)
    intMin = ctypes.c_double(0)
    gaussFit = ctypes.c_ushort(2)
    sigma0 = ctypes.c_double(1)
    theta0 = ctypes.c_double(0)
    size_ROI_fit_c = ctypes.c_ushort(size_ROI_fit)

    dll._OpenPALMProcessing(img_ptr, buff_ptr, potential_points, H, W,
                            waveletNo, thresholdVal, watershedRatio,
                            volMin, intMin,
                            gaussFit, sigma0, sigma0, theta0,
                            size_ROI_fit_c)
    _ = dll._PALMProcessing()
    dll._closePALMProcessing()

    pts = np.ctypeslib.as_array(buff_ptr, shape=(potential_points,))

    locs = []
    for k in range(0, len(pts), 13):
        cx = pts[k + 4]
        cy = pts[k + 3]         # PALM order: [..., y, x, ...]
        locs.append((cx, cy))
        # breaking condition
        if (cx, cy) in [(0.0, 0.0), (-1.0, -1.0)]:
            break

    return locs



def build_paired_roi_stacks_batch(
    paths_high: Sequence[str],
    paths_low: Sequence[str],
    paths_mask: Optional[Sequence[str]] = None, 
    *,
    n_patches: Optional[int] = None,
    t_window: int = 5,
    threshold: float = 180.0,
    size_ROI_fit: int = 7,
    size_ROI_crop: int = 8,
    border_margin: int = 5,
    dll_path: str = "./SCOL/CPU_PALM.dll",
    seed: Optional[int] = None,
    output_prefix: str = "batch",
    ratio_loc: float = 0.5,
    mask_box_size: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # Ajout mask_box_size
    """
    Extract paired ROIs from high and low SNR stacks.
    ...
    mask_box_size (int): Size of the protective square in the binary mask (e.g., 5 for 5x5). Must be an odd integer.
    """
 
    assert t_window % 2 == 1, "t_window must be odd"
    margin_t = t_window // 2 
    mask_radius = mask_box_size // 2
    rng = np.random.default_rng(seed)

    dll = ctypes.cdll.LoadLibrary(dll_path)
    potential_points = 49999
    buff_ptr = np.zeros((potential_points,), dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    rois_high, rois_low, rois_mask = [], [], []

    for i in range(len(paths_high)):
        stack_A = tifffile.imread(paths_high[i])
        stack_B = tifffile.imread(paths_low[i])
        n_frames, H, W = stack_A.shape

        # Détection globale pour le masque et la cible synthétique
        locs_by_frame = {}
        all_loc_candidates = []
        print(f"Stack {i+1}: Détection PALM...")
        for f in range(margin_t, n_frames - margin_t):
            locs = _detect_points_PALM(stack_A[f], dll, buff_ptr, potential_points, H, W, threshold, size_ROI_fit)
            locs_by_frame[f] = locs
            for cx, cy in locs:
                all_loc_candidates.append((f, cx, cy))

        # Quotas
        target_loc = min(int(n_patches * ratio_loc) // len(paths_high), len(all_loc_candidates))
        target_bg = (n_patches // len(paths_high)) - target_loc
        
        rng.shuffle(all_loc_candidates)
        selected = [('loc', f, cx, cy) for f, cx, cy in all_loc_candidates[:target_loc]]
        
        # Ajout de background
        for _ in range(target_bg):
            f_bg = rng.integers(margin_t, n_frames - margin_t)
            # On prend un point aléatoire (loin des bords)
            rx = rng.integers(border_margin + size_ROI_crop//2, W - border_margin - size_ROI_crop//2)
            ry = rng.integers(border_margin + size_ROI_crop//2, H - border_margin - size_ROI_crop//2)
            selected.append(('bg', f_bg, rx, ry))

        rng.shuffle(selected)

        # Extraction et génération synthétique
        for c_type, f, cx, cy in selected:
            # Jitter
            dx, dy = (rng.integers(-8, 9), rng.integers(-8, 9)) if c_type == 'loc' else (0,0)
            x_c, y_c = int(round(cx + dx)), int(round(cy + dy))
            x_min, x_max = x_c - size_ROI_crop//2, x_c + size_ROI_crop//2
            y_min, y_max = y_c - size_ROI_crop//2, y_c + size_ROI_crop//2

            if x_min < 0 or y_min < 0 or x_max > W or y_max > H: continue

            # --- INPUT LOW (Réel) ---
            rB = stack_B[f - margin_t : f + margin_t + 1, y_min:y_max, x_min:x_max]
            if rB.shape[-1] != size_ROI_crop: continue

            # --- TARGET HIGH (Synthétique) & MASK ---
            rA_syn = np.zeros((1, size_ROI_crop, size_ROI_crop), dtype=np.float32)
            rM = np.ones((1, size_ROI_crop, size_ROI_crop), dtype=np.float32)
            
            y_grid, x_grid = np.indices((size_ROI_crop, size_ROI_crop))
            
            for mx, my in locs_by_frame.get(f, []):
                if x_min <= mx < x_max and y_min <= my < y_max:
                    lx, ly = mx - x_min, my - y_min
                    # Masque
                    li, lj = int(round(ly)), int(round(lx))
                    rM[0, max(0, li-mask_radius):li+mask_radius+1, max(0, lj-mask_radius):lj+mask_radius+1] = 0
                    # Gaussienne idéale (Sigma 1.0)
                    rA_syn[0] += np.exp(-((x_grid - lx)**2 + (y_grid - ly)**2) / (2 * 1.8**2))
                    
            rA_syn = np.clip(rA_syn, 0.0, 1.0)
            rois_high.append(np.clip(rA_syn, 0, 1))
            rois_low.append(rB)
            rois_mask.append(rM)

    # --- FINALISATION (BIEN INDENTÉE) ---
    if len(rois_high) == 0:
        raise ValueError("Aucun patch n'a été extrait. Vérifiez vos chemins ou votre seuil.")
        
    s_high = np.stack(rois_high)
    s_low  = np.stack(rois_low)
    s_mask = np.stack(rois_mask)

    perm = rng.permutation(len(s_high))
    s_high, s_low, s_mask = s_high[perm], s_low[perm], s_mask[perm]

    tifffile.imwrite(f"{output_prefix}_high.tif", s_high)
    tifffile.imwrite(f"{output_prefix}_low.tif", s_low)
    tifffile.imwrite(f"{output_prefix}_mask.tif", s_mask)

    return s_high, s_low, s_mask



paths_high = [r"data/NUP/NUP_high.tif"]
paths_low = [r"data/NUP/NUP_low_shifted.tif"]
paths_mask = None #[r"nup_seuil26_cell.tif"]

t0 = time.time()
build_paired_roi_stacks_batch(
    paths_high, 
    paths_low,
    paths_mask,
    n_patches=20000,
    t_window=1,
    threshold=17.9,
    size_ROI_fit=7,
    size_ROI_crop=64,
    border_margin=10,
    seed=42,
    output_prefix="100",
    ratio_loc=1,
    mask_box_size=5
)
print(f"Finished in {time.time() - t0:.2f} second(s).")