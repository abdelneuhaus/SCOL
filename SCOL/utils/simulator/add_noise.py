import numpy as np

def add_noise(image_to_noised, bckg, sd):

    noisy = np.random.poisson(image_to_noised)
    noisy = np.array(noisy, dtype='float64')
    noisy += np.random.normal(loc=bckg, scale=sd, size=image_to_noised.shape)
    noisy[noisy < 0] = bckg
    return np.array(noisy, dtype='uint16')
