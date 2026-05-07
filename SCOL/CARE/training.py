import matplotlib.pyplot as plt

from csbdeep.utils import axes_dict, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE



def do_training(kernel_size:int=5, unet_depth:int=2, unet_first_layer:int=64, train_epochs:int=40, 
                lr=4e-4, steps_epochs:int=400, data_care:str="model.npz", save_in:str="current_directory"):
    
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

    config = Config(axes=axes,
                    probabilistic=False,
                    n_channel_in=3,
                    n_channel_out=1,
                    unet_kern_size=kernel_size,
                    unet_n_depth=unet_depth,
                    unet_n_first=unet_first_layer,
                    train_batch_size=16,
                    # train_steps_per_epoch=100,  # 400 initially
                    train_epochs=train_epochs,
                    train_learning_rate=lr,
                    unet_residual=True,
                    train_reduce_lr = {'factor':0.5, 'patience':10, 'min_delta':1e-4},
                    train_loss = 'weighted_loss_tf',
                    )

    print(config)
    vars(config)
    model = CARE(config, save_in, basedir='models')
    model.keras_model.summary()

    history = model.train(X,Y, validation_data=(X_val,Y_val))

    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
