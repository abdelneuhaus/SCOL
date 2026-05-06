import matplotlib.pyplot as plt

from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE



def show_some_data(X_val, Y_val):    
        plt.figure(figsize=(12,5))
        plot_some(X_val[:5],Y_val[:5])
        plt.suptitle('Example validation patches (top: source, bottom: target)')
        plt.show()


def do_training(kernel_size=5, unet_depth=2, unet_first_layer=16, train_epochs=40,
                lr=4e-4, steps_epochs=100, data_care="none.npz", save_in="here"):

    (X,Y), (X_val,Y_val), axes = load_training_data(data_care, validation_split=0.1, verbose=True)
    c = axes_dict(axes)['Z']

    config = Config(axes=axes,
                    probabilistic=False,
                    n_channel_in=3,
                    n_channel_out=1,
                    unet_kern_size=kernel_size,
                    unet_n_depth=unet_depth,
                    unet_n_first=64,    # or 32
                    train_batch_size=16,
                    # train_steps_per_epoch=100,  # 400 initially
                    train_epochs=train_epochs,
                    train_learning_rate=lr,
                    unet_residual=False,
                    train_reduce_lr = {'factor':0.5, 'patience':10, 'min_delta':1e-4},
                    train_loss = 'weighted_loss_tf',   # weighted_loss_tf
                    )

    print(config)
    vars(config)
    model = CARE(config, save_in, basedir='models')
    model.keras_model.summary()

    history = model.train(X,Y, validation_data=(X_val,Y_val))

    print(sorted(list(history.history.keys())))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])

    # plt.figure(figsize=(20,12))
    # _P = model.keras_model.predict(X_val[:5])
    # if config.probabilistic:
    #     _P = _P[...,:(_P.shape[-1]//2)]
    #     plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
    #     plt.suptitle('5 example validation patches\n'      
    #                 'top row: input (source), '          
    #                 'middle row: target (ground truth), '
    #                 'bottom row: predicted from source')
    # plt.show()
