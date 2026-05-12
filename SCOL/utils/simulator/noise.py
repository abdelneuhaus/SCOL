import numpy as np

from typing import Union

def add_noise(image_to_noised:np.ndarray, background:Union[float, int], sd:Union[float, int]):
    """
    Add simple mixture of Poisson and Gaussian noises to an image containing only PSFs.

    Args:
        image_to_noised (np.ndarray): trained model
        background (float|int): path to low SNR image.
        sd (float|int): path to high SNR image.

    Returns:
        np.ndarray corresponding to the image with pseudocamera noise.
    """
    
    noisy = np.random.poisson(image_to_noised).astype(np.float64)
    noisy += np.random.normal(loc=background, scale=sd, size=image_to_noised.shape)
    noisy[noisy < 0] = background
    return noisy.astype(np.uint16)


