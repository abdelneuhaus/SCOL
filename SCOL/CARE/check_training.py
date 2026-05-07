import os
import numpy as np
import matplotlib.pyplot as plt
import time as time

from tifffile import imread
from csbdeep.utils import plot_some
from csbdeep.models import CARE



def check_training_3D(network:str, low_path:str, high_path:str, n:int=3):
    """
    Check training model using a pair of low and high images.

    Args:
        network: trained model
        low_path (str): path to low SNR image.
        high_path (str): path to high SNR image.
        n (int): time window.

    Returns:
        None. Displays a matplotlib plot.
    """
    t_start = 10    # frame to display
    mid = n // 2    # get the middle frame from the set of t_window frames
    
    # load set of t_window frames
    x_raw = imread(low_path)[t_start : t_start + n]
    y_gt = imread(high_path)[t_start + mid]
    x_input = np.moveaxis(x_raw, 0, -1) # Go from (n, Y, X) to (Y, X, n)
    x_ready = x_input.astype(np.float32)

    axes = 'YXC'
    model = CARE(config=None, name=network, basedir='')
    restored = model.predict(x_ready, axes, normalizer=None)
    
    # middle frame extraction
    n_out = restored.shape[-1]  # how many frames the network has produced
    if n_out == 1:
        # N-to-1 model
        pred_center = restored[..., 0]
    else:
        # N-to-N model
        pred_center = restored[..., mid] 
    low_center = x_input[..., mid]
    
    # plotting
    plt.figure(figsize=(15,6))
    to_plot = np.stack([low_center, pred_center, y_gt])
    plot_some(to_plot,
              title_list=[['Low Input (Center)', 'Prediction (Center)', 'GT (Center)']], 
              pmin=0.1, pmax=99.9)
    
    plt.suptitle(f"Checking training of the following model: {network} ({n}-to-{n_out})")
    plt.show()



start= time.time()

low_path = 'data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/noisy.tif'
high_path = 'data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/target.tif'
check_training_3D('C:/Git/SCOL/models/data/SIMU/TEST/models/k3_d1_f64', n=3)

end = time.time()
print((end-start)/60, "minutes")
