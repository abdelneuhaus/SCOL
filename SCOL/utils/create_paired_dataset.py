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
    t_window_high: int = 5,
    t_window_low: int = 5,
    threshold: float = 180.0,
    size_ROI_fit: int = 7,
    size_ROI_crop: int = 8,
    border_margin: int = 5,
    dll_path: str = "./SCOL/CPU_PALM.dll",
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
 
    # Assertions basiques
    assert t_window_high % 2 == 1, "t_window_high must be odd (eg: 1, 3, 5, 7)"
    assert t_window_low % 2 == 1, "t_window_low must be odd (eg: 1, 3, 5, 7)"
    assert 0 <= ratio_loc <= 1, "ratio_loc must be between 0 and 1"
    assert len(paths_high) == len(paths_low), "paths_high and paths_low must have the same length."

    margin_t_high = t_window_high // 2 
    margin_t_low = t_window_low // 2 

    rng = np.random.default_rng(seed)
    if seed is not None:
        random.seed(seed)

    # Initialisation de la DLL
    dll = ctypes.cdll.LoadLibrary(dll_path)
    potential_points = 49999
    empty_buff = np.zeros((potential_points,), dtype=np.float64)
    buff_ptr = empty_buff.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    rois_high, rois_low = [], []

    # Si plusieurs stacks sont donnés, on divise le budget de patchs entre eux
    n_patches_per_stack = n_patches // len(paths_high) if n_patches is not None else None

    # Boucle sur les paires d'images
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

        # =========================================================
        # PASSAGE 1 : DÉTECTION (On récolte toutes les coordonnées)
        # =========================================================
        print(f"Stack {i+1}/{len(paths_high)}: Recherche globale des molécules...")
        for f in range(margin_t_low, n_frames - margin_t_low):
            frameA_center = stack_A[f] 
            locs = _detect_points_PALM(frameA_center, dll, buff_ptr,
                                       potential_points, H, W,
                                       threshold, size_ROI_fit)
            for cx, cy in locs:
                if (min_center_x <= cx <= max_center_x and min_center_y <= cy <= max_center_y):
                    if mask[int(cy), int(cx)] > 0:
                        all_loc_candidates.append((f, cx, cy))

        total_locs_found = len(all_loc_candidates)
        print(f"Total des molécules potentielles détectées : {total_locs_found}")

        # =========================================================
        # CALCUL DES QUOTAS 
        # =========================================================
        if n_patches_per_stack is not None:
            target_loc = int(n_patches_per_stack * ratio_loc)
            target_bg = n_patches_per_stack - target_loc
        else:
            # Comportement par défaut si n_patches=None
            target_loc = total_locs_found
            target_bg = int(target_loc * (1 - ratio_loc) / max(ratio_loc, 1e-6)) if ratio_loc < 1 else 0

        # Si l'image contient moins de molécules que ce qu'on demande
        if target_loc > total_locs_found:
            print(f"Attention: {target_loc} patchs molécules demandés, mais seulement {total_locs_found} trouvés.")
            target_loc = total_locs_found
            # On ajuste le background pour conserver le ratio de 65/35 exact
            if ratio_loc > 0:
                target_bg = int(target_loc * (1 - ratio_loc) / ratio_loc)

        print(f"Objectif d'extraction : {target_loc} patchs Molécules, {target_bg} patchs Fond.")

        # =========================================================
        # SÉLECTION ALÉATOIRE
        # =========================================================
        rng.shuffle(all_loc_candidates)
        selected_locs = all_loc_candidates[:target_loc]

        selected_bgs = []
        if len(valid_x) > 0 and target_bg > 0:
            for _ in range(target_bg):
                f_bg = rng.integers(margin_t_low, n_frames - margin_t_low)
                idx = rng.integers(0, len(valid_x))
                selected_bgs.append((f_bg, valid_x[idx], valid_y[idx]))

        # =========================================================
        # PASSAGE 2 : EXTRACTION DES PIXELS
        # =========================================================
        # On regroupe les cibles (Molécule = 'loc', Fond = 'bg')
        crops_to_extract = [('loc', f, cx, cy) for f, cx, cy in selected_locs] + \
                           [('bg', f, cx, cy) for f, cx, cy in selected_bgs]
        
        # On remélange pour que le Dataset final ne soit pas ordonné
        rng.shuffle(crops_to_extract)

        for c_type, f, cx, cy in crops_to_extract:
            if c_type == 'loc':
                # Jitter (léger décalage aléatoire pour la molécule)
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
                # Background standard (sans jitter supplémentaire)
                x_min, x_max = int(cx) - size_ROI_crop // 2, int(cx) + size_ROI_crop // 2
                y_min, y_max = int(cy) - size_ROI_crop // 2, int(cy) + size_ROI_crop // 2

            # Extraction et sauvegarde dans les buffers
            rA = stack_A[f - margin_t_high : f + margin_t_high + 1, y_min:y_max, x_min:x_max]
            rB = stack_B[f - margin_t_low : f + margin_t_low + 1, y_min:y_max, x_min:x_max]
            rois_high.append(rA)
            rois_low.append(rB)

    # Conversion finale
    if not rois_high:
        raise RuntimeError("Aucune ROI valide trouvée.")

    stack_high = np.stack(rois_high) 
    stack_low  = np.stack(rois_low)  

    # Dernier mélange de sécurité
    perm = rng.permutation(stack_high.shape[0])
    stack_high = stack_high[perm]
    stack_low  = stack_low[perm]

    outA = f"{output_prefix}_high_ROIs.tif"
    outB = f"{output_prefix}_low_ROIs.tif"
    tifffile.imwrite(outA, stack_high.astype(np.float32), imagej=True)
    tifffile.imwrite(outB, stack_low.astype(np.float32), imagej=True)

    print(f"{stack_high.shape[0]} ROIs enregistrées dans {outA} et {outB}")
    return stack_high, stack_low



paths_high = [r"C:/Git/SCOL/data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/high.tif"]
paths_low = [r"C:/Git/SCOL/data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/low.tif"]
paths_mask = None #[r"nup_seuil26_cell.tif"]

t0 = time.time()
build_paired_roi_stacks_batch(
    paths_high, 
    paths_low,
    paths_mask,
    n_patches=5000,
    t_window_high=3,
    t_window_low=3,
    threshold=24,
    size_ROI_fit=8,
    size_ROI_crop=64,
    border_margin=5,
    seed=41,
    output_prefix="75",
    ratio_loc=0.9
)
print(f"Finished in {time.time() - t0:.2f} second(s).")