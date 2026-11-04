import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from csbdeep.models import Config, CARE
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_history



def _git_info():
    """
    Return the current commit and the modified (not committed) files of the SCOL repository.

    Returns:
        dict: commit and modified files, or None if git is not available.
    """
    repo = Path(__file__).resolve().parent
    try:
        git = lambda *args: subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
        return {'commit': git('rev-parse', 'HEAD'),
                'modified_files': git('status', '--porcelain', '--untracked-files=no').splitlines()}
    except Exception:
        return None


def _write_json(path, content):
    """Write a dict to a JSON file."""
    with open(path, 'w') as f:
        json.dump(content, f, indent=2)


def do_training(kernel_size:int=5, unet_depth:int=2, unet_first_layer:int=64, train_epochs:int=40, 
                lr=4e-4, data_care:str="model.npz", data_val:str="model_val.npz", save_in:str="current_directory",
                loss_choice='MAE', seed:int=0, dataset_dir:str=None):
    """
    Perform the training using 2D-CARE network.
    Based on https://github.com/CSBDeep/CSBDeep code, specifically tutorial notebooks.

    A run_info.json file is written in the model folder: git commit, data, parameters, seed,
    dataset generation parameters, and the best epoch at the end of the training.

    Args:
        kernel_size (int): kernel size (generally 3x3 or 5x5). Must be odd.
        unet_depth (int): U-Net depth.
        unet_first_layer (int): number of layers in the first layer
        train_epochs (int): number of epochs to train the network
        lr (float): initial learning rate value.
        steps_epochs (int): number of steps per epochs
        data_care (str): path to NPZ file (training data).
        data_val (str): path to NPZ file (validation data, split by do_data_processing).
        save_in (str): path to save the trained model.
        loss_choice (str): loss function. Only 'MAE' (CSBDeep standard) for now.
        seed (int): random seed (weights initialisation and batch order).
        dataset_dir (str, optional): "Training" folder; its dataset_info.json is copied into run_info.json.
    Returns:
        None. Save the 2D-CARE model and display losses curves.
    """
    tf.keras.utils.set_random_seed(seed)    # also sets the Python and NumPy seeds

    (X,Y), _, axes = load_training_data(data_care, validation_split=0, verbose=True)
    (X_val,Y_val), _, _ = load_training_data(data_val, validation_split=0, verbose=True)
    c = axes_dict(axes)['Z']

    n_chan_in = X.shape[-1] if len(X.shape) == 4 else 1
    n_chan_out = Y.shape[-1] if len(Y.shape) == 4 else 1

    config = Config(axes=axes,
                    probabilistic=False,
                    n_channel_in=n_chan_in,
                    n_channel_out=n_chan_out,
                    unet_kern_size=kernel_size,
                    unet_n_depth=unet_depth,
                    unet_n_first=unet_first_layer,
                    train_batch_size=16,
                    train_epochs=train_epochs,
                    train_learning_rate=lr,
                    unet_residual=False,
                    train_reduce_lr = {'factor':0.5, 'patience':10, 'min_delta':1e-4}
                    )

    print(config)
    vars(config)
    model = CARE(config, save_in, basedir='models')

    model.prepare_for_training()

    assert loss_choice == 'MAE', f"Unknown loss: {loss_choice}"
    print("Using MAE loss (CSBDeep standard).")

    # traceability: what produced this model
    info_path = Path(dataset_dir) / 'dataset_info.json' if dataset_dir else None
    run_info = {
        'date': datetime.now().isoformat(timespec='seconds'),
        'git': _git_info(),
        'loss': loss_choice,
        'seed': seed,
        't_window_low': n_chan_in,
        't_window_high': n_chan_out,
        'hyperparameters': {'kernel_size': kernel_size, 'unet_depth': unet_depth, 'unet_first_layer': unet_first_layer,
                            'train_epochs': train_epochs, 'lr': lr, 'batch_size': 16},
        'data_train': {'path': str(data_care), 'n_patches': len(X), 'augmented': 'augmented' in np.load(data_care).files},
        'data_val': {'path': str(data_val), 'n_patches': len(X_val)},
        'dataset_info': json.load(open(info_path)) if info_path and info_path.exists() else None,
    }
    _write_json(model.logdir / 'run_info.json', run_info)
    print(f"Run info saved in {model.logdir / 'run_info.json'}")

    model.keras_model.summary()
    history = model.train(X,Y, validation_data=(X_val,Y_val))

    # best epoch (the one saved in weights_best.h5)
    val_loss = history.history['val_loss']
    run_info['best_epoch'] = int(np.argmin(val_loss)) + 1
    run_info['best_val_loss'] = float(np.min(val_loss))
    _write_json(model.logdir / 'run_info.json', run_info)

    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])