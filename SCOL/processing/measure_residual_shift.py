"""Measure the residual misalignment between a LOW movie and its registered HIGH movie.

Both movies must be on the same pixel grid (HIGH already warped onto LOW) and have
the same number of frames: use exactly the two files that were given to the patch
generator. Spots are detected in each frame of both movies (wavelet plane 2, as in
PALMTracer), localized with a 3-point Gaussian fit, and paired by mutual nearest
neighbour. Offsets are dx, dy = HIGH - LOW, in pixels.

The script reports:
    * the median offset over all pairs (global translation);
    * the median offset per zone of the field (residual that varies in space);
    * an affine fit HIGH = f(LOW) and the offset it predicts at the centre and
      at the four corners of the field;
    * the number of pairs for a frame shift of -1, 0, +1 (frame mismatch check).

Usage: edit the parameters below, then run ``python measure_registration_residual.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- parameters
LOW_PATH = "data/SPT/low_Cell01.tif"
HIGH_PATH = "data/SPT/high_Cell01_shifted.tif"
N_FRAMES = 1000     # frames analysed, spread evenly over the movie
N_SIGMA = 6.0       # detection threshold, in units of wavelet noise (PALMTracer ~3.5)
MAX_DIST = 3.0      # px, maximum distance between the two spots of a pair
BORDER = 4          # px ignored along the edges (must be >= 1)
N_ZONES = 4         # the field is split into N_ZONES x N_ZONES zones
PIXEL_NM = 160.0
# ---------------------------------------------------------------------------

_K1 = np.array([1, 4, 6, 4, 1], np.float32) / 16
_K2 = np.array([1, 0, 4, 0, 6, 0, 4, 0, 1], np.float32) / 16


def _open(path):
    """Opens a TIFF stack without loading it in memory when possible."""
    try:
        return tifffile.memmap(path, mode="r")
    except Exception:
        return tifffile.imread(path)


def _smooth(img, kernel):
    out = ndimage.convolve1d(img, kernel, axis=0, mode="mirror")
    return ndimage.convolve1d(out, kernel, axis=1, mode="mirror")


def wavelet_plane2(img):
    """Second plane of the a trous B3 wavelet transform (PALMTracer detection)."""
    a1 = _smooth(img, _K1)
    return a1 - _smooth(a1, _K2)


def wavelet_noise(movie, frames):
    """Noise of the wavelet plane 2, estimated with the MAD over a few frames."""
    values = np.concatenate([wavelet_plane2(np.asarray(movie[f], np.float32)).ravel()
                             for f in frames[:20]])
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def _vertex(m, c, p):
    """Sub-pixel peak position from three samples.

    Gaussian fit (parabola on the log), or a plain parabola when a sample is <= 0.
    """
    ok = (m > 0) & (c > 0) & (p > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lm, lc, lp = np.log(m), np.log(c), np.log(p)
        gauss = 0.5 * (lm - lp) / (lm - 2 * lc + lp)
        parab = 0.5 * (m - p) / (m - 2 * c + p)
    offset = np.where(ok, gauss, parab)
    return np.clip(np.nan_to_num(offset), -0.5, 0.5)


def detect(img, threshold):
    """Returns the (x, y) positions of the spots of one frame."""
    w = wavelet_plane2(np.asarray(img, np.float32))
    peaks = (w == ndimage.maximum_filter(w, size=3)) & (w > threshold)
    peaks[:BORDER, :] = False
    peaks[-BORDER:, :] = False
    peaks[:, :BORDER] = False
    peaks[:, -BORDER:] = False
    ys, xs = np.nonzero(peaks)
    dx = _vertex(w[ys, xs - 1], w[ys, xs], w[ys, xs + 1])
    dy = _vertex(w[ys - 1, xs], w[ys, xs], w[ys + 1, xs])
    return np.column_stack([xs + dx, ys + dy])


def mutual_pairs(a, b, max_dist):
    """Pairs of points of a and b that are each other's nearest neighbour."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    dist_ab, idx_ab = cKDTree(b).query(a)
    _, idx_ba = cKDTree(a).query(b)
    keep = (dist_ab < max_dist) & (idx_ba[idx_ab] == np.arange(len(a)))
    return a[keep], b[idx_ab[keep]]


def fit_affine(src, dst, n_iter=3):
    """Least-squares affine fit dst = [x, y, 1] @ A, with outlier rejection."""
    design = np.column_stack([src, np.ones(len(src))])
    keep = np.ones(len(src), bool)
    for _ in range(n_iter):
        A = np.linalg.lstsq(design[keep], dst[keep], rcond=None)[0]
        residual = np.linalg.norm(design @ A - dst, axis=1)
        keep = residual < 2.5 * np.median(residual[keep])
    return A, keep


def main():
    low, high = _open(LOW_PATH), _open(HIGH_PATH)
    assert low.shape == high.shape, f"LOW {low.shape} and HIGH {high.shape} differ"
    n_frames, height, width = low.shape
    frames = np.unique(np.linspace(0, n_frames - 1, min(N_FRAMES, n_frames)).astype(int))

    thr_low = N_SIGMA * wavelet_noise(low, frames)
    thr_high = N_SIGMA * wavelet_noise(high, frames)
    print(f"Movies: {n_frames} frames of {height} x {width} px, {len(frames)} analysed")
    print(f"Thresholds: LOW {thr_low:.1f}, HIGH {thr_high:.1f} ({N_SIGMA:g} x wavelet noise)")

    cache = {}

    def spots(movie, name, f):
        if (name, f) not in cache:
            cache[(name, f)] = detect(movie[f], thr_low if name == "low" else thr_high)
        return cache[(name, f)]

    # Frame mismatch check: HIGH at t against LOW at t-1, t, t+1.
    check = [f for f in frames if 0 < f < n_frames - 1][:100]
    counts = {k: sum(len(mutual_pairs(spots(high, "high", f), spots(low, "low", f + k),
                                      MAX_DIST)[0]) for f in check)
              for k in (-1, 0, 1)}
    print(f"\nPairs for a frame shift of -1 / 0 / +1: "
          f"{counts[-1]} / {counts[0]} / {counts[1]} (0 must be clearly the highest)")

    # Pairs over all analysed frames.
    p_low, p_high = [], []
    for f in frames:
        a, b = mutual_pairs(spots(low, "low", f), spots(high, "high", f), MAX_DIST)
        p_low.append(a)
        p_high.append(b)
    p_low, p_high = np.concatenate(p_low), np.concatenate(p_high)
    d = p_high - p_low
    med = np.median(d, axis=0)
    rsd = 1.4826 * np.median(np.abs(d - med), axis=0)
    print(f"\n{len(d)} pairs (dx, dy = HIGH - LOW)")
    print(f"Median offset: dx = {med[0]:+.3f} px ({med[0] * PIXEL_NM:+.0f} nm), "
          f"dy = {med[1]:+.3f} px ({med[1] * PIXEL_NM:+.0f} nm)")
    print(f"Spread (robust sd): {rsd[0]:.2f} / {rsd[1]:.2f} px "
          f"(mostly localization noise, not registration)")
    if np.any(np.abs(med) > 0.6 * MAX_DIST):
        print("WARNING: median close to MAX_DIST, increase MAX_DIST and run again.")

    # Median offset per zone.
    zx = np.clip((p_low[:, 0] / width * N_ZONES).astype(int), 0, N_ZONES - 1)
    zy = np.clip((p_low[:, 1] / height * N_ZONES).astype(int), 0, N_ZONES - 1)
    zone_med = np.full((N_ZONES, N_ZONES, 2), np.nan)
    print(f"\nMedian offset per zone (dx, dy in px), top row = top of the image:")
    for j in range(N_ZONES):
        row = []
        for i in range(N_ZONES):
            sel = (zx == i) & (zy == j)
            if sel.sum() >= 30:
                zone_med[j, i] = np.median(d[sel], axis=0)
                row.append(f"({zone_med[j, i, 0]:+.2f}, {zone_med[j, i, 1]:+.2f})")
            else:
                row.append("    (few pairs)")
        print("  " + "  ".join(row))

    # Affine fit and offset predicted across the field.
    A, keep = fit_affine(p_low, p_high)
    print(f"\nAffine fit ({keep.sum()} pairs kept): "
          f"scale x {A[0, 0]:.5f}, scale y {A[1, 1]:.5f}, "
          f"cross terms {A[1, 0]:+.5f} / {A[0, 1]:+.5f}")
    points = {"centre": (width / 2, height / 2),
              "top left": (BORDER, BORDER), "top right": (width - BORDER, BORDER),
              "bottom left": (BORDER, height - BORDER),
              "bottom right": (width - BORDER, height - BORDER)}
    worst = 0.0
    for name, (x, y) in points.items():
        off = np.array([x, y, 1.0]) @ A - np.array([x, y])
        worst = max(worst, float(np.hypot(*off)))
        print(f"  {name:>12}: dx {off[0]:+.2f}, dy {off[1]:+.2f} px")

    print(f"\nLargest offset in the field: {worst:.2f} px ({worst * PIXEL_NM:.0f} nm)")
    if worst < 0.1:
        print("Verdict: registration OK, it does not explain the wider spots.")
    elif worst < 0.2:
        print("Verdict: borderline, may widen the spots slightly.")
    else:
        print("Verdict: residual too large, fix the registration and regenerate Training_L5.")

    # Figure: histograms and zone map.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.arange(-MAX_DIST, MAX_DIST + 0.05, 0.05)
    for ax, k, label in zip(axes[:2], (0, 1), ("dx", "dy")):
        ax.hist(d[:, k], bins=bins, color="0.4")
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(med[k], color="r", lw=1.2, label=f"median {med[k]:+.3f} px")
        ax.set_xlabel(f"{label} = HIGH - LOW (px)")
        ax.legend()
    gain = 100
    cx = (np.arange(N_ZONES) + 0.5) * width / N_ZONES
    cy = (np.arange(N_ZONES) + 0.5) * height / N_ZONES
    gx, gy = np.meshgrid(cx, cy)
    axes[2].quiver(gx, gy, zone_med[..., 0] * gain, zone_med[..., 1] * gain,
                   angles="xy", scale_units="xy", scale=1, color="r")
    axes[2].set_xlim(0, width)
    axes[2].set_ylim(height, 0)
    axes[2].set_aspect("equal")
    axes[2].set_title(f"Median offset per zone (arrows x{gain})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()