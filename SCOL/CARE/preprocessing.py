import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tifffile import imread
from csbdeep.utils import plot_some



def do_data_processing(basepath:str, save_path:str="", t_window_low:int=3, t_window_high:int=3, simulation:bool=False):
    """
    Prepare the .npz data for CARE training. 
    Adapted from https://github.com/CSBDeep/CSBDeep code. 

    Args:
        basepath (str): path to NPZ file (training data).
        savepath (str): path to save the NPZ file.
        t_window_low (int): number of consecutive frames to use to create noisy training data (X)
        t_window_high (int): number of consecutive frames to use to create target training data (Y)
        simulation (bool): if True, ignore temporal window (1-to-1). To use for simulation images only.

    Returns:
        None. Save the NPZ file and show some training paired dataset.
    """

    base = Path(basepath)  
    low_files = list((base / 'Low').glob('*.tif*'))
    high_files = list((base / 'High').glob('*.tif*'))
    
    assert len(low_files) > 0, "No TIF file found in the 'Low' repertory."
    assert len(high_files) > 0, "No TIF file found in the 'High' repertory."
    
    print("Loading data...")
    X = imread(str(low_files[0])).astype(np.float32)
    Y = imread(str(high_files[0])).astype(np.float32)

    # handling of dimension (for simulation)
    if simulation:
        if X.ndim == 3:
            X = np.expand_dims(X, axis=1)
        if Y.ndim == 3:
            Y = np.expand_dims(Y, axis=1)
    else:
        # default
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        if Y.ndim == 3:
            Y = np.expand_dims(Y, axis=0)

    print(f"\nTemporal extraction: {t_window_low} frame(s) for X, {t_window_high} frame(s) for Y.")
    
    total_frames_x = X.shape[1]
    total_frames_y = Y.shape[1]
    center_x = total_frames_x // 2
    center_y = total_frames_y // 2
    start_x = center_x - (t_window_low // 2)
    end_x = start_x + t_window_low
    start_y = center_y - (t_window_high // 2)
    end_y = start_y + t_window_high

    # check that ROIs dont go over the border of image
    if start_x < 0 or end_x > total_frames_x:
        raise ValueError(f"Impossible to extract {t_window_low} frames. This axis only contained {total_frames_x} elements.")
    if start_y < 0 or end_y > total_frames_y:
        raise ValueError(f"Impossible to extract {t_window_high} frames. This axis only contained  {total_frames_y} elements.")

    X = X[:, start_x:end_x, :, :]
    Y = Y[:, start_y:end_y, :, :]

    if save_path == "":
        save_path = str(basepath).replace('Training/', 'model.npz')
    else:
        save_path += "_model.npz"
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    axes = 'SCYX' 
    np.savez(save_path, X=X, Y=Y, axes=axes)
    print("\nSave in :", save_path)
    print("Shape of X =", X.shape)
    print("Shape of Y =", Y.shape)
    print("Axes       =", axes)
    
    # display some training data
    idx_plot_x = X.shape[1] // 2
    idx_plot_y = Y.shape[1] // 2
    
    for i in range(min(2, max(1, X.shape[0] // 8))):
        plt.figure(figsize=(16,4))
        sl = slice(8*i, 8*(i+1))
        X_plot = X[sl, idx_plot_x]
        Y_plot = Y[sl, idx_plot_y]
        try:
            plot_some(X_plot, Y_plot, title_list=[np.arange(sl.start, sl.stop)])
            plt.show()
        except NameError:
            print("plot_some not found. Skipping the visualization")
            break



def inspect_channels_in_npz(npz_path, roi_index=0):
    """
    Display one training sample by showing t_window images. 

    Args:
        npz_path (str): path to NPZ file (training data).
        roi_index (str): sample index.

    Returns:
        None. Show one training sample.
    """
    data = np.load(npz_path)
    X = data['X'] # X=low, Y=high
    print(f"Shape of dataset : {X.shape}")
    
    if X.ndim != 4:
        print("Dataset doesnt have 4 dimensions (SCYX).")
        return
        
    S, C, Y, X_dim = X.shape
    print(f"ROI {roi_index} contained {C} channels (frames).")
    roi = X[roi_index]
    
    plt.figure(figsize=(4 * C, 4))
    for c in range(C):
        plt.subplot(1, C, c + 1)
        plt.imshow(roi[c], cmap='magma')
        plt.title(f"Channel {c}")
        plt.axis('off')
        
    plt.suptitle(f"ROI #{roi_index}", fontsize=14)
    plt.tight_layout()
    plt.show()
