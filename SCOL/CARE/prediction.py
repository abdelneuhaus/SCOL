import numpy as np
import time as time
from tqdm import tqdm

from tifffile import imread
from csbdeep.models import CARE
from csbdeep.io import save_tiff_imagej_compatible



def do_denoising_3D_fast(input_path, output_path, model_name, basedir='C:/Git/SCOL/models/data/SIMU/TEST/models'):
    """
    
    """
    print(f"Chargement du modèle: {model_name}")
    model = CARE(config=None, name=model_name, basedir=basedir)
    
    # --- AUTO-DÉTECTION DE L'ARCHITECTURE ---
    n_in = model.config.n_channel_in
    n_out = model.config.n_channel_out
    print(f"Architecture détectée : {n_in}-to-{n_out}")

    print(f"Chargement du stack: {input_path}")
    stack = imread(input_path) # Attend (T, Y, X)
    
    # 1. Padding Temporel Dynamique
    pad = n_in // 2
    if pad > 0:
        stack_padded = np.pad(stack, ((pad, pad), (0,0), (0,0)), mode='edge')
    else:
        stack_padded = stack
        
    stack_padded_ready = stack_padded.astype(np.float32)
    n_timepoints = stack.shape[0]
    restored_stack = np.zeros(stack.shape, dtype=np.float32)
    print("Début du débruitage temporel (sans normalisation)...")
    
    # 2. Boucle de Prédiction Image par Image
    for t in tqdm(range(n_timepoints)):
        # Extraire la fenêtre temporelle
        window = stack_padded_ready[t : t + n_in] # (n_in, Y, X)
        window_yxc = np.moveaxis(window, 0, -1)   # (Y, X, C)
        
        # --- PRÉDICTION CSBDEEP ---
        # Le normalizer=None rend l'opération quasi instantanée sur GPU.
        # Si vous avez une erreur de RAM, ajoutez n_tiles=(2, 2, 1) ici.
        restored_img = model.predict(window_yxc, axes='YXC', normalizer=None)
        
        # 3. Extraction Intelligente
        if n_out == 1:
            pred_center = restored_img[..., 0] # Prend le seul canal de sortie
        elif n_out == n_in:
            pred_center = restored_img[..., pad] # Prend le canal central
        else:
            raise ValueError(f"Format de sortie inconnu : {n_in} entrées pour {n_out} sorties.")
            
        # Assigner l'image dans le stack final
        restored_stack[t] = pred_center

    # 4. Sauvegarde
    print(f"Sauvegarde en 32-bits dans {output_path}...")
    save_tiff_imagej_compatible(output_path, restored_stack, axes='TYX')
    print("Terminé.")
    


start= time.time()
model_path = "C:/Git/SCOL/models/data/SIMU/TEST/models"
do_denoising_3D_fast(input_path="data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/noisy.tif", 
                     output_path="data/GROUND_TRUTH/SIMULATION/metrics_simu/fixed/noisy_weighted5.tif", 
                     model_name="k3_d1_f64")
end = time.time()
print((end-start)/60, "minutes")