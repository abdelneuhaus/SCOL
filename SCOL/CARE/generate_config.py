import os
from pathlib import Path

def generate_training_config(data_path="", save_in="", kernel_size=3, unet_depth=1, unet_first_layer=64, train_epochs=40, lr=1e-4):
    """
    Génère la configuration d'entraînement et crée un dossier unique auto-incrémenté 
    incluant la signature de l'architecture (ex: training_k3d1l64_0001).
    """
    
    model_directory = Path(data_path).parent / 'models'
    model_directory.mkdir(parents=True, exist_ok=True)
    hyper_sig = f"k{kernel_size}d{unet_depth}l{unet_first_layer}"
    
    counter = 1
    while True:
        training_folder_name = f"training_{hyper_sig}_{counter:04d}"
        training_output_directory = model_directory / training_folder_name
        
        if not training_output_directory.exists():
            break
        counter += 1

    model_path = f"{save_in}_model.npz"

    config = {
        "kernel_size": kernel_size,
        "unet_depth": unet_depth,
        "unet_first_layer": unet_first_layer,
        "train_epochs": train_epochs,
        "lr": lr,
        "data_care": model_path,
        "save_in": str(training_output_directory)
    }
    
    print(f"Saving in : {training_output_directory.name}")
    return config