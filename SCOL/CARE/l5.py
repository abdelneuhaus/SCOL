"""
Test of the L5 loss: MAE weighted by localization masks (symmetric version).

Experimental, next to training.py (not modified). The HIGH patches come with a mask: squares around
the PALMTracer detections (processing/create_paired_dataset_mask.py). During training, the loss also
finds the spots drawn by the network (wavelet filter, same threshold as PALMTracer on the HIGH) and
weights the errors:

    W  = 1 + w * (target squares OR predicted squares) + lam * (predicted squares NOT in target squares)
    L5 = mean(W * |prediction - HIGH|)

Missing a molecule costs about (1 + w) times more than with the MAE, inventing one (1 + w + lam) times
more. w = 0, lam = 0 gives the standard MAE (reference, same code).
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tifffile import imread

from csbdeep.models import Config, CARE
from csbdeep.io import load_training_data
from csbdeep.utils import plot_history

from care.training import _git_info, _write_json


# ---------------------------------------------------------------- data

def do_data_processing_l5(dataset_dir: str, save_path: str, t_window_low: int = 3,
                          val_fraction: float = 0.1, seed: int = 0) -> tuple:
    """
    Build the training and validation .npz files, with the mask as a second channel of Y.

    Args:
        dataset_dir (str): folder with the Low, High and Mask sub-folders (one TIF file each).
        save_path (str): prefix of the .npz files (<save_path>_model.npz and <save_path>_model_val.npz).
        t_window_low (int): number of central LOW frames to keep (odd).
        val_fraction (float): fraction of the patches drawn at random for validation (0.05 to 0.15).
        seed (int): random seed of the split.

    Returns:
        tuple: paths of the training and validation .npz files.
    """
    base = Path(dataset_dir)
    files = {}
    for sub in ("Low", "High", "Mask"):
        found = list((base / sub).glob("*.tif*"))
        assert len(found) == 1, f"Expected one TIF file in '{sub}', found {len(found)}: {[f.name for f in found]}"
        files[sub] = found[0]
    print("Loading", ", ".join(f.name for f in files.values()), "...")

    X, Y, M = (imread(str(files[s])).astype(np.float32) for s in ("Low", "High", "Mask"))
    X, Y, M = (a[:, np.newaxis] if a.ndim == 3 else a for a in (X, Y, M))   # ImageJ TIFs drop axes of size 1
    assert X.shape[0] == Y.shape[0] == M.shape[0], "Low, High and Mask do not have the same number of patches."
    assert Y.shape[1] == 1 and M.shape[1] == 1, "High and Mask must have one frame per patch."

    n_frames = X.shape[1]
    assert t_window_low % 2 == 1 and t_window_low <= n_frames, f"t_window_low must be odd and <= {n_frames}."
    start = n_frames // 2 - t_window_low // 2
    X = X[:, start:start + t_window_low]                  # central LOW frames
    Y = np.concatenate([Y, M], axis=1)                    # channel 0: HIGH image, channel 1: mask (read by the loss only)

    assert 0.05 <= val_fraction <= 0.15, "val_fraction must be between 0.05 and 0.15."
    idx = np.random.default_rng(seed).permutation(len(X))
    n_val = round(val_fraction * len(X))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_path, val_path = Path(f"{save_path}_model.npz"), Path(f"{save_path}_model_val.npz")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(train_path, X=X[train_idx], Y=Y[train_idx], axes="SCYX")
    np.savez(val_path, X=X[val_idx], Y=Y[val_idx], axes="SCYX")
    print(f"Training   : {train_path} | X {X[train_idx].shape} | Y {Y[train_idx].shape} (image + mask)")
    print(f"Validation : {val_path} | X {X[val_idx].shape} | Y {Y[val_idx].shape}")
    return str(train_path), str(val_path)


# ---------------------------------------------------------------- spot detection (TensorFlow)

_K1 = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16
_K2 = np.array([1, 0, 4, 0, 6, 0, 4, 0, 1], dtype=np.float32) / 16


def _separable_conv(img, k):
    """Separable convolution of a (B, Y, X, 1) tensor with the 1D kernel k, mirror borders."""
    r = len(k) // 2
    x = tf.pad(img, [[0, 0], [r, r], [r, r], [0, 0]], mode="REFLECT")
    x = tf.nn.conv2d(x, tf.constant(k.reshape(-1, 1, 1, 1)), strides=1, padding="VALID")
    return tf.nn.conv2d(x, tf.constant(k.reshape(1, -1, 1, 1)), strides=1, padding="VALID")


def wavelet_plane2(img):
    """Second plane of the 'a trous' wavelet transform (B3 spline), the usual SMLM detection filter."""
    a1 = _separable_conv(img, _K1)
    a2 = _separable_conv(a1, _K2)
    return a1 - a2


def _spots(img, threshold, border=4):
    """
    Spots of a (B, Y, X, 1) image: local maxima of the wavelet plane above threshold (in ADU).

    The band of `border` pixels at the patch border is ignored (the filter is not reliable there).
    """
    w = wavelet_plane2(img)
    peaks = tf.cast(tf.equal(w, tf.nn.max_pool2d(w, 3, 1, "SAME")) & (w > threshold), tf.float32)
    inner = tf.pad(tf.ones_like(peaks[:, border:-border, border:-border, :]),
                   [[0, 0], [border, border], [border, border], [0, 0]])
    return tf.stop_gradient(peaks * inner)


def _predicted_squares(y_pred, threshold, mask_size):
    """Squares of side mask_size around the spots drawn by the network (no gradient)."""
    return tf.stop_gradient(tf.nn.max_pool2d(_spots(y_pred, threshold), mask_size, 1, "SAME"))


# ---------------------------------------------------------------- loss and metrics

def _split_mask(y_true):
    """Mask channel of y_true -> (molecule squares, ghost sites). Values: 1 = molecule at t, 2 = ghost site, 0 = background."""
    mask = y_true[..., 1:2]
    return tf.cast(tf.equal(mask, 1.0), tf.float32), tf.cast(tf.equal(mask, 2.0), tf.float32)


def loss_l5(w: float = 10.0, lam: float = 0.0, threshold: float = 200.0, mask_size: int = 7, w_ghost: float = 0.0):
    """
    L5 loss (symmetric version). y_true[..., 0] is the HIGH image, y_true[..., 1] the target mask
    (1 = molecule at t, 2 = ghost site, 0 = background).

    Args:
        w (float): extra weight of the errors in the target and predicted squares.
        lam (float): extra weight of the errors in the predicted squares without a target square (invented spots).
        threshold (float): wavelet threshold (ADU) to find the spots drawn by the network; same as PALMTracer on the HIGH.
        mask_size (int): side of the squares, same as the target mask.
        w_ghost (float): extra weight of the errors in the ghost sites (molecule at t - 1 or t + 1, none at t).
            w_ghost = w: a ghost costs as much as a missed molecule.

    Returns:
        function: Keras loss.
    """
    def l5(y_true, y_pred):
        y = y_true[..., 0:1]
        m, g = _split_mask(y_true)
        pm = _predicted_squares(y_pred, threshold, mask_size)
        W = 1.0 + w * tf.maximum(m, pm) + lam * pm * (1.0 - m) + w_ghost * g
        return tf.reduce_mean(W * tf.abs(y_pred - y))
    return l5


def mae_img(y_true, y_pred):
    """MAE on the HIGH image (same as the standard loss): comparable between all runs."""
    return tf.reduce_mean(tf.abs(y_pred - y_true[..., 0:1]))


def mse_img(y_true, y_pred):
    """MSE on the HIGH image."""
    return tf.reduce_mean(tf.square(y_pred - y_true[..., 0:1]))


def mae_mol(y_true, y_pred):
    """MAE on the pixels of the target squares only (how well the molecules are drawn)."""
    m, _ = _split_mask(y_true)
    return tf.reduce_sum(m * tf.abs(y_pred - y_true[..., 0:1])) / (tf.reduce_sum(m) + 1.0)


def mae_ghost(y_true, y_pred):
    """MAE on the pixels of the ghost sites only (faint ghosts included): lower = fewer ghosts."""
    _, g = _split_mask(y_true)
    return tf.reduce_sum(g * tf.abs(y_pred - y_true[..., 0:1])) / (tf.reduce_sum(g) + 1.0)


def fp_spots_metric(threshold: float = 200.0):
    """Metric: number of spots drawn by the network outside the target squares, per patch."""
    def fp_spots(y_true, y_pred):
        m, _ = _split_mask(y_true)
        return tf.reduce_sum(_spots(y_pred, threshold) * (1.0 - m)) / tf.cast(tf.shape(y_pred)[0], tf.float32)
    return fp_spots


# ---------------------------------------------------------------- test without training

def test_loss_l5(w: float = 10.0, lam: float = 10.0, threshold: float = 200.0, mask_size: int = 7, seed: int = 0):
    """
    Check the loss on a simulated patch: perfect prediction, missed molecule, invented spot.

    Prints the extra cost of each case compared with the perfect prediction, for the MAE and for L5.
    """
    rng = np.random.default_rng(seed)
    P = 64
    yy, xx = np.mgrid[0:P, 0:P]
    spot = lambda r, c, a=1500: a * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / 2)
    clean = 1440 + spot(31, 31)
    target = clean + rng.normal(0, 203, (P, P))
    m = np.zeros((P, P)); m[28:35, 28:35] = 1
    y_true = tf.constant(np.stack([target, m], -1)[None], tf.float32)
    as_tensor = lambda a: tf.constant(a[None, ..., None], tf.float32)
    preds = {"perfect": clean, "missed molecule": np.full((P, P), 1440.0), "invented spot": clean + spot(12, 48)}

    ref, l5 = loss_l5(0, 0, threshold, mask_size), loss_l5(w, lam, threshold, mask_size)
    p0 = as_tensor(preds["perfect"])
    print(f"w = {w:g}, lam = {lam:g}: extra cost compared with the perfect prediction")
    for name, p in preds.items():
        p = as_tensor(p)
        d_ref = float(ref(y_true, p) - ref(y_true, p0))
        d_l5 = float(l5(y_true, p) - l5(y_true, p0))
        ratio = f"x{d_l5 / d_ref:.1f}" if d_ref > 0 else ""
        print(f"  {name:16s}  MAE +{d_ref:6.2f}   L5 +{d_l5:7.2f}   {ratio}")


# ---------------------------------------------------------------- training

def do_training_l5(data_train: str, data_val: str, models_root: str, kernel_size: int = 3, unet_depth: int = 2,
                   unet_first_layer: int = 64, train_epochs: int = 100, lr: float = 1e-4, w: float = 10.0,
                   lam: float = 0.0, threshold: float = 200.0, mask_size: int = 7, w_ghost: float = 0.0,
                   seed: int = 0, dataset_dir: str = None):
    """
    Train a 2D-CARE network (N-to-1, one output) with the L5 loss.

    The model folder is models_root/L5_w<w>_lam<lam>_k<k>d<d>l<f>_T<T>_<n>, with run_info.json.
    The model is a standard CARE model: notebook 3 (prediction) works without change.

    Args:
        data_train (str): training .npz (Y with 2 channels: image + mask).
        data_val (str): validation .npz.
        models_root (str): folder holding the models.
        kernel_size, unet_depth, unet_first_layer: U-Net architecture.
        train_epochs (int): number of epochs.
        lr (float): initial learning rate.
        w, lam, threshold, mask_size, w_ghost: L5 parameters (see loss_l5). w = lam = w_ghost = 0: standard MAE.
        seed (int): random seed.
        dataset_dir (str, optional): dataset folder; its dataset_info.json is copied into run_info.json.

    Returns:
        History: Keras training history.
    """
    tf.keras.utils.set_random_seed(seed)
    (X, Y), _, axes = load_training_data(data_train, validation_split=0, verbose=True)
    (X_val, Y_val), _, _ = load_training_data(data_val, validation_split=0, verbose=True)
    assert Y.shape[-1] == 2, "Y must have 2 channels (HIGH image + mask): use do_data_processing_l5."
    T = X.shape[-1]

    signature = f"L5_w{w:g}_lam{lam:g}_g{w_ghost:g}_k{kernel_size}d{unet_depth}l{unet_first_layer}_T{T}"
    n = 1
    while (Path(models_root) / f"{signature}_{n:04d}").exists():
        n += 1

    config = Config(axes=axes, probabilistic=False, n_channel_in=T, n_channel_out=1,
                    unet_kern_size=kernel_size, unet_n_depth=unet_depth, unet_n_first=unet_first_layer,
                    train_batch_size=16, train_epochs=train_epochs, train_learning_rate=lr,
                    unet_residual=False, train_tensorboard=False,
                    train_reduce_lr={'factor': 0.5, 'patience': 10, 'min_delta': 1e-4})
    print(config)
    model = CARE(config, f"{signature}_{n:04d}", basedir=str(models_root))
    model.prepare_for_training()
    model.keras_model.compile(optimizer=model.keras_model.optimizer,
                              loss=loss_l5(w, lam, threshold, mask_size, w_ghost),
                              metrics=[mae_img, mse_img, mae_mol, mae_ghost, fp_spots_metric(threshold)])

    info_path = Path(dataset_dir) / "dataset_info.json" if dataset_dir else None
    run_info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "loss": "L5" if (w or lam or w_ghost) else "MAE (L5 code, w = lam = w_ghost = 0)",
        "loss_params": {"w": w, "lam": lam, "w_ghost": w_ghost, "threshold": threshold, "mask_size": mask_size},
        "seed": seed,
        "t_window_low": T, "t_window_high": 1,
        "hyperparameters": {"kernel_size": kernel_size, "unet_depth": unet_depth, "unet_first_layer": unet_first_layer,
                            "train_epochs": train_epochs, "lr": lr, "batch_size": 16},
        "data_train": {"path": str(data_train), "n_patches": len(X), "augmented": "augmented" in np.load(data_train).files},
        "data_val": {"path": str(data_val), "n_patches": len(X_val)},
        "dataset_info": json.load(open(info_path)) if info_path and info_path.exists() else None,
    }
    _write_json(model.logdir / "run_info.json", run_info)
    print(f"Run info saved in {model.logdir / 'run_info.json'}")

    history = model.train(X, Y, validation_data=(X_val, Y_val))

    val_loss = history.history["val_loss"]
    run_info["best_epoch"] = int(np.argmin(val_loss)) + 1
    run_info["best_val_loss"] = float(np.min(val_loss))
    run_info["final_val_metrics"] = {k: float(v[-1]) for k, v in history.history.items() if k.startswith("val_")}
    _write_json(model.logdir / "run_info.json", run_info)

    plt.figure(figsize=(16, 8))
    plot_history(history, ["loss", "val_loss"], ["mae_img", "val_mae_img"],
                 ["mae_mol", "val_mae_mol"], ["mae_ghost", "val_mae_ghost"], ["fp_spots", "val_fp_spots"])
    plt.show()
    return history