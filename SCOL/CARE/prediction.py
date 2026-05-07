import numpy as np
import time as time
from tqdm import tqdm

from tifffile import imread
from csbdeep.models import CARE
from csbdeep.io import save_tiff_imagej_compatible



def do_denoising_3D_fast(input_path, output_path, model_name, basedir='C:/Git/SCOL/models/data/SIMU/TEST/models'):
    """
    Perfom restoration of noisy image.

    Args:
        input_path (str): path to noisy image.
        output_path (str): path to save restored image.
        model_name (str): model name.
        basedir (str): path where the model is saved.

    Returns:
        None. Save the restored image. 
    """
    print(f"Chargement du modèle: {model_name}")
    model = CARE(config=None, name=model_name, basedir=basedir)
    
    n_in = model.config.n_channel_in
    n_out = model.config.n_channel_out
    print(f"2.5D time CARE configuration: {n_in}-to-{n_out}")
    print(f"Stack loading: {input_path}")
    stack = imread(input_path) # need (T, Y, X)
    
    # temporal padding
    pad = n_in // 2
    if pad > 0:
        stack_padded = np.pad(stack, ((pad, pad), (0,0), (0,0)), mode='edge')
    else:
        stack_padded = stack
        
    stack_padded_ready = stack_padded.astype(np.float32)
    n_timepoints = stack.shape[0]
    restored_stack = np.zeros(stack.shape, dtype=np.float32)
    print("Denoising starting...")
    
    # prediction loop (image one by one)
    for t in tqdm(range(n_timepoints)):
        window = stack_padded_ready[t : t + n_in] # (n_in, Y, X)
        window_yxc = np.moveaxis(window, 0, -1)   # (Y, X, C)
        
        # actual prediction
        # normalizer=None make it instataneous on GPU. if RAM error, add n_tiles=(2, 2, 1) here.
        restored_img = model.predict(window_yxc, axes='YXC', normalizer=None)
        if n_out == 1:
            pred_center = restored_img[..., 0] # take only one channel
        elif n_out == n_in:
            pred_center = restored_img[..., pad] # take middle channel
        else:
            raise ValueError(f"Format de sortie inconnu : {n_in} entrées pour {n_out} sorties.")
            
        restored_stack[t] = pred_center

    print(f"Saving in 32-bits in {output_path}...")
    save_tiff_imagej_compatible(output_path, restored_stack, axes='TYX')
    print("Restoration finished.")
    


start= time.time()
model_path = "C:/Git/SCOL/models/data/SIMU/TEST/models"
do_denoising_3D_fast(input_path="data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/noisy.tif", 
                     output_path="data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/noisy_weighted5.tif", 
                     model_name="k3_d1_f64")
end = time.time()
print((end-start)/60, "minutes")