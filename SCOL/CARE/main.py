from preprocessing import do_data_processing, inspect_channels_in_npz
from data_augmentation import apply_data_augmentation_to_npz
from training import do_training


full_image = 'data/SIMU/TEST/Training'
base_save_in = str(full_image).replace('/Training', '/models/')

# hyperparameters grid
kernel_size = 3
unet_depth = 1
unet_first_layer = 64
list_config = []

# build configuration
model_name = f"k{kernel_size}_d{unet_depth}_f{unet_first_layer}"
save_in = f"{base_save_in}{model_name}"

config = {
    "kernel_size": kernel_size,
    "unet_depth": unet_depth,
    "unet_first_layer": unet_first_layer,
    "train_epochs": 40,
    "lr": 1e-4,
    "data_care": save_in,  # model access path
    "save_in": save_in  # dossier sorti du training
}

list_config.append(config)

# Run experiments
for c in list_config:
    print(f"Training model: {c['save_in']}")

    # do_data_processing(full_image , save_path=c["save_in"], t_window_high=3, t_window_low=3, simulation=False)
    c['data_care'] += '_model.npz'
    # apply_data_augmentation_to_npz(c['data_care'])
    # inspect_channels_in_npz(c['data_care'], 2200)

    # c['data_care'] += '_model.npz'
    do_training(kernel_size=c['kernel_size'], unet_depth=c['unet_depth'], unet_first_layer=c['unet_first_layer'], 
                train_epochs=c['train_epochs'], lr=c['lr'], data_care=c['data_care'], save_in=c['save_in'])