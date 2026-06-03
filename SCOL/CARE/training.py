import matplotlib.pyplot as plt

from care.custom_losses import loss_custom_spt_data, loss_custom_fixed_data, loss_custom_fixed_data_v2

from csbdeep.models import Config, CARE
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, plot_history



def do_training(kernel_size:int=5, unet_depth:int=2, unet_first_layer:int=64, train_epochs:int=40, 
                lr=4e-4, data_care:str="model.npz", save_in:str="current_directory",
                loss_choice='MSE Standard (CSBDeep)', loss_params=None):
    
    """
    Perform the training using 2D-CARE network.
    Based on https://github.com/CSBDeep/CSBDeep code, specifically tutorial notebooks.

    Args:
        kernel_size (int): kernel size (generally 3x3 or 5x5). Must be odd.
        unet_depth (int): U-Net depth.
        unet_first_layer (int): number of layers in the first layer
        train_epochs (int): number of epochs to train the network
        lr (float): initial learning rate value.
        steps_epochs (int): number of steps per epochs
        data_care (str): path to NPZ file
        save_in (str): path to save the trained model.
    Returns:
        None. Save the 2D-CARE model and display losses curves.
    """

    (X,Y), (X_val,Y_val), axes = load_training_data(data_care, validation_split=0.1, verbose=True)
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


    if loss_choice == 'MSE Standard (CSBDeep)':
        print("Using MSE loss.")
        pass 
        

    elif loss_choice == 'Custom SSIM+MSE (Fixed Data)':
        gamma = loss_params['gamma'] if loss_params else 2.0
        alpha = loss_params['alpha'] if loss_params else 0.65
        model.keras_model.compile(optimizer=model.keras_model.optimizer, 
                                  loss=loss_custom_fixed_data(gamma=gamma, alpha=alpha), 
                                  metrics=['mse', 'mae'])
        
    elif loss_choice == 'Custom SPT Data':
        spt_val = loss_params['spt_val'] if loss_params else 20        
        model.keras_model.compile(optimizer=model.keras_model.optimizer, 
                                  loss=loss_custom_spt_data(spt_val), 
                                  metrics=['mse', 'mae'])

    model.keras_model.summary()
    history = model.train(X,Y, validation_data=(X_val,Y_val))

    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
