import numpy as np


def apply_data_augmentation_to_npz(npz_path:str):
    """
    Perfom data augmentation (8 times) of training images contained in NPZ file format.

    Args:
        npz_path (str): path to NPZ file (training data).

    Returns:
        None. Save the new NPZ file
    """
    print(f"\n--- Beginning of data augmentation of {npz_path} ---")
    
    data = np.load(npz_path)
    X, Y = data['X'], data['Y']
    axes_str = str(data['axes']) if data['axes'].shape == () else str(data['axes'][0])
    X_aug, Y_aug = [], []
    
    for k in range(4):
        # rotate by 0°, 90°, 180°, 270°
        X_rot = np.rot90(X, k=k, axes=(2, 3))
        Y_rot = np.rot90(Y, k=k, axes=(2, 3))
        X_aug.append(X_rot)
        Y_aug.append(Y_rot)
        # vertical flip
        X_aug.append(np.flip(X_rot, axis=2))
        Y_aug.append(np.flip(Y_rot, axis=2))
        
    X_final = np.concatenate(X_aug, axis=0)
    Y_final = np.concatenate(Y_aug, axis=0)
    
    # shuffle to avoid having the same data following one another
    perm = np.random.permutation(X_final.shape[0])
    X_final = X_final[perm]
    Y_final = Y_final[perm]
    
    # 4. Sauvegarde par-dessus le fichier existant
    np.savez(npz_path, X=X_final, Y=Y_final, axes=axes_str)
    
    print(f"Finshed : dataset now have {X_final.shape[0]} ROIs.")
