import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from csbdeep.utils import plot_some
from pathlib import Path

def show_paired_data(training_path, test_path):
    y = imread(training_path)
    x = imread(test_path)
    print('image size =', x.shape)

    # Gestion de la temporalité : extraction de la frame centrale pour l'affichage
    if x.ndim == 4: # Format (N_rois, Canaux, Y, X)
        idx_c = x.shape[1] // 2
        x_show = x[0, idx_c]
        y_show = y[0, idx_c] if y.shape[1] > 1 else y[0, 0]
    elif x.ndim == 3: # Format (Canaux, Y, X)
        idx_c = x.shape[0] // 2
        x_show = x[idx_c]
        y_show = y[idx_c] if y.shape[0] > 1 else y[0]
    else:
        x_show, y_show = x, y

    plt.figure(figsize=(13,5))
    plt.subplot(1,2,1)
    plt.imshow(x_show, cmap="magma")
    plt.colorbar()
    plt.title("Low (Frame centrale)")
    
    plt.subplot(1,2,2)
    plt.imshow(y_show, cmap="magma")
    plt.colorbar()
    plt.title("High (Frame centrale)")
    plt.show()



def robust_scale(data, p_low=0.1, p_high=99.9):
    """
    Applique un Robust Scaling sur les données.
    Calcule les centiles pour ignorer les valeurs extrêmes (ex: hot pixels),
    normalise entre 0 et 1, et coupe ce qui dépasse.
    """
    print(f"  Calcul des centiles {p_low}% et {p_high}%...")
    vmin = np.percentile(data, p_low)
    vmax = np.percentile(data, p_high)
    
    print(f"  -> Valeur min (bruit de fond) estimée : {vmin:.2f}")
    print(f"  -> Valeur max (sommet PSF) estimée : {vmax:.2f}")
    
    # Normalisation avec protection contre la division par zéro
    scaled = (data - vmin) / (vmax - vmin + 1e-8)
    
    # Clipping : les hot pixels (> vmax) sont ramenés à 1.0, le bruit extrême (< vmin) à 0.0
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)





# def do_data_processing(basepath, save_path="", t_window_low=3, t_window_high=1):
#     """
#     Prépare les données .npz pour l'entraînement CARE.
    
#     Paramètres:
#     - t_window_low (int): Nombre de frames temporelles à conserver pour l'image d'entrée Low SNR (X).
#     - t_window_high (int): Nombre de frames temporelles à conserver pour la cible High SNR (Y).
#     """
#     base = Path(basepath)
    
#     low_files = list((base / 'Low').glob('*.tif*'))
#     high_files = list((base / 'High').glob('*.tif*'))
    
#     assert len(low_files) > 0, "Aucun fichier TIF trouvé dans le dossier Low."
#     assert len(high_files) > 0, "Aucun fichier TIF trouvé dans le dossier High."
    
#     print("Chargement des images...")
#     X = imread(str(low_files[0])).astype(np.float32)
#     Y = imread(str(high_files[0])).astype(np.float32)

#     # --- Gestion temporelle via t_window ---
#     print(f"\nExtraction temporelle : {t_window_low} frame(s) pour X, {t_window_high} frame(s) pour Y.")
    
#     total_frames_x = X.shape[1]
#     total_frames_y = Y.shape[1]
    
#     center_x = total_frames_x // 2
#     center_y = total_frames_y // 2
    
#     # Calcul des indices autour du centre
#     start_x = center_x - (t_window_low // 2)
#     end_x = start_x + t_window_low
    
#     start_y = center_y - (t_window_high // 2)
#     end_y = start_y + t_window_high

#     # Sécurités : On s'assure que les fenêtres demandées ne débordent pas de l'image
#     if start_x < 0 or end_x > total_frames_x:
#         raise ValueError(f"Impossible d'extraire {t_window_low} frames. X ne contient que {total_frames_x} frames.")
#     if start_y < 0 or end_y > total_frames_y:
#         raise ValueError(f"Impossible d'extraire {t_window_high} frames. Y ne contient que {total_frames_y} frames.")

#     # Découpage dynamique
#     X = X[:, start_x:end_x, :, :]
#     Y = Y[:, start_y:end_y, :, :]
#     # ---------------------------------------

#     if save_path == "":
#         save_path = str(basepath).replace('Training/', 'model.npz')
#     else:
#         save_path += "_model.npz"
        
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     axes = 'SCYX' 
#     np.savez(save_path, X=X, Y=Y, axes=axes)
    
#     print("\nSauvegarde réussie :", save_path)
#     print("Shape of X =", X.shape)
#     print("Shape of Y =", Y.shape)
#     print("Axes       =", axes)
    
#     # --- Visualisation robuste aux dimensions asymétriques ---
#     idx_plot_x = X.shape[1] // 2
#     idx_plot_y = Y.shape[1] // 2
    
#     for i in range(min(2, max(1, X.shape[0] // 8))):
#         plt.figure(figsize=(16,4))
#         sl = slice(8*i, 8*(i+1))
        
#         X_plot = X[sl, idx_plot_x]
#         Y_plot = Y[sl, idx_plot_y]
        
#         try:
#             plot_some(X_plot, Y_plot, title_list=[np.arange(sl.start, sl.stop)])
#             plt.show()
#         except NameError:
#             print("Fonction 'plot_some' non trouvée, saut de la visualisation.")
#             break




def do_data_processing(basepath, save_path="", t_window_low=3, t_window_high=1, simulation=False):
    """
    Prépare les données .npz pour l'entraînement CARE.
    
    Paramètres:
    - t_window_low (int): Nombre de frames temporelles à conserver pour l'image d'entrée Low SNR (X).
    - t_window_high (int): Nombre de frames temporelles à conserver pour la cible High SNR (Y).
    - simulation (bool): Si True, traite les images 3D (XYT) de façon 1-to-1 en ignorant 
                         la notion de fenêtre temporelle (format SCYX où S=T et C=1).
    """
    base = Path(basepath)
    
    low_files = list((base / 'Low').glob('*.tif*'))
    high_files = list((base / 'High').glob('*.tif*'))
    
    assert len(low_files) > 0, "Aucun fichier TIF trouvé dans le dossier Low."
    assert len(high_files) > 0, "Aucun fichier TIF trouvé dans le dossier High."
    
    print("Chargement des images...")
    X = imread(str(low_files[0])).astype(np.float32)
    Y = imread(str(high_files[0])).astype(np.float32)

    # --- AJOUT: GESTION DES DIMENSIONS POUR LA SIMULATION ---
    if simulation:
        # En mode simulation CARE standard, on transforme XYT (T, Y, X) en SCYX (S, 1, Y, X)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=1)
        if Y.ndim == 3:
            Y = np.expand_dims(Y, axis=1)
    else:
        # Comportement par défaut : s'il manque l'axe ROI (S), on l'ajoute au début (1, T, Y, X)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        if Y.ndim == 3:
            Y = np.expand_dims(Y, axis=0)

    # --- Gestion temporelle via t_window ---
    print(f"\nExtraction temporelle : {t_window_low} frame(s) pour X, {t_window_high} frame(s) pour Y.")
    
    total_frames_x = X.shape[1]
    total_frames_y = Y.shape[1]
    
    center_x = total_frames_x // 2
    center_y = total_frames_y // 2
    
    # Calcul des indices autour du centre
    start_x = center_x - (t_window_low // 2)
    end_x = start_x + t_window_low
    
    start_y = center_y - (t_window_high // 2)
    end_y = start_y + t_window_high

    # Sécurités : On s'assure que les fenêtres demandées ne débordent pas de l'image
    if start_x < 0 or end_x > total_frames_x:
        raise ValueError(f"Impossible d'extraire {t_window_low} frames. L'axe ciblé ne contient que {total_frames_x} éléments.")
    if start_y < 0 or end_y > total_frames_y:
        raise ValueError(f"Impossible d'extraire {t_window_high} frames. L'axe ciblé ne contient que {total_frames_y} éléments.")

    # Découpage dynamique (Maintenant sécurisé car l'array est garanti 4D)
    X = X[:, start_x:end_x, :, :]
    Y = Y[:, start_y:end_y, :, :]
    # ---------------------------------------

    if save_path == "":
        save_path = str(basepath).replace('Training/', 'model.npz')
    else:
        save_path += "_model.npz"
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    axes = 'SCYX' 
    np.savez(save_path, X=X, Y=Y, axes=axes)
    
    print("\nSauvegarde réussie :", save_path)
    print("Shape of X =", X.shape)
    print("Shape of Y =", Y.shape)
    print("Axes       =", axes)
    
    # --- Visualisation robuste aux dimensions asymétriques ---
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
            print("Fonction 'plot_some' non trouvée, saut de la visualisation.")
            break



def inspect_channels_in_npz(npz_path, roi_index=0):
    """
    Ouvre le fichier .npz et affiche les frames (channels) d'une seule ROI
    pour vérifier la cohérence spatio-temporelle.
    """
    data = np.load(npz_path)
    X = data['X'] # Y = high
    
    print(f"Shape du dataset X : {X.shape}")
    
    if X.ndim != 4:
        print("ERREUR : X n'a pas 4 dimensions (SCYX).")
        return
        
    S, C, Y, X_dim = X.shape
    print(f"La ROI {roi_index} contient {C} channels (frames).")
    
    # Extraction d'une ROI spécifique
    roi = X[roi_index]
    
    plt.figure(figsize=(4 * C, 4))
    for c in range(C):
        plt.subplot(1, C, c + 1)
        plt.imshow(roi[c], cmap='magma')
        plt.title(f"Channel (Frame) {c}")
        plt.axis('off')
        
    plt.suptitle(f"ROI n°{roi_index} - Séquence temporelle", fontsize=14)
    plt.tight_layout()
    plt.show()


def apply_data_augmentation_to_npz(npz_path):
    """
    Ouvre le fichier d'entraînement, multiplie les données par 8 via 
    rotations et symétries, mélange le tout, et sauvegarde.
    """
    print(f"\n--- Début de la Data Augmentation sur {npz_path} ---")
    
    # 1. Chargement des données brutes
    data = np.load(npz_path)
    X, Y = data['X'], data['Y']
    
    # L'axe de 'axes' stocké dans un .npz ressort parfois comme un objet 0-d
    axes_str = str(data['axes']) if data['axes'].shape == () else str(data['axes'][0])
    
    # On sait que l'axe est 'SCYX', donc l'axe Y est 2 et l'axe X est 3.
    X_aug, Y_aug = [], []
    
    for k in range(4):
        # Rotations à 0°, 90°, 180°, 270°
        X_rot = np.rot90(X, k=k, axes=(2, 3))
        Y_rot = np.rot90(Y, k=k, axes=(2, 3))
        
        X_aug.append(X_rot)
        Y_aug.append(Y_rot)
        
        # Effet Miroir (Flip vertical) appliqué sur chaque rotation
        X_aug.append(np.flip(X_rot, axis=2))
        Y_aug.append(np.flip(Y_rot, axis=2))
        
    # 2. Concaténation (On passe de N à N*8)
    X_final = np.concatenate(X_aug, axis=0)
    Y_final = np.concatenate(Y_aug, axis=0)
    
    # 3. Mélange (Shuffle) indispensable pour que le réseau ne voie 
    # pas les 8 versions de la même molécule à la suite
    print("Mélange aléatoire des données (Shuffle)...")
    perm = np.random.permutation(X_final.shape[0])
    X_final = X_final[perm]
    Y_final = Y_final[perm]
    
    # 4. Sauvegarde par-dessus le fichier existant
    np.savez(npz_path, X=X_final, Y=Y_final, axes=axes_str)
    
    print(f"Terminé : Le dataset est passé de {X.shape[0]} à {X_final.shape[0]} ROIs.")
    print("---------------------------------------------------\n")