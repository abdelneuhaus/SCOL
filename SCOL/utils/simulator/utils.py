import numpy as np
import random

def generate_intensity(value=1400, sd=0):
    """

    """
    return int(np.random.randint(value, sd))


def save_parameters(filename, frames, nb_emitters, intensity, length_min, length_max, blink_min, blink_max, background_value, sd_bckg_value, edge):
    """
    
    """
    filename = filename.replace('.tif','')
    with open(str(filename)+'_parameters.txt', 'w') as f:
        f.write('Number of frames: '+ str(frames))
        f.write('\n')
        f.write('Numbers of emitters: ' + str(nb_emitters))
        f.write('\n')
        f.write('Integrated Intensity: ' +str(intensity))
        f.write('\n')
        f.write('Background value: ' +str(background_value)+', sd: '+str(sd_bckg_value))
        f.write('\n')        
        f.write('ON duration: ['+ str(length_min)+', '+str(length_max)+']')
        f.write('\n')
        f.write('Number of blinks: ['+ str(blink_min)+', '+str(blink_max)+']')
        f.write('\n')



def generate_on_times(frames, randomize=True, off_length_min=1, off_length_max=3, number_blink_min=1, number_blink_max=3, beads=False):
    """
    
    """
    number_blink = random.choice(range(number_blink_min, number_blink_max+1))
    if number_blink_max == number_blink_min:
        number_blink = number_blink_min
    blink = []
    if beads and frames > 1:
        a = random.choice(range(0, int(frames*0.1)))
        b = random.choice(range(int(frames*0.9), frames))
        return list(range(a,b))
    if randomize:
        for i in range(number_blink):
            off_length = random.choice(range(off_length_min, off_length_max+1))
            a = random.randint(0, frames-off_length)
            blink += list(range(a, a+off_length))
        return sorted(set(blink))
    else:
        return list(range(frames-10, frames+1))
    


def add_noise(image_to_noised:np.ndarray, background:float|int, sd:float|int):
    """
    Add simple mixture of Poisson and Gaussian noises to an image containing only PSFs.

    Args:
        image_to_noised (np.ndarray): trained model
        background (float|int): path to low SNR image.
        sd (float|int): path to high SNR image.

    Returns:
        np.ndarray corresponding to the image with pseudocamera noise.
    """
    noisy = np.random.poisson(image_to_noised)
    noisy = np.array(noisy, dtype='float64')
    noisy += np.random.normal(loc=background, scale=sd, size=image_to_noised.shape)
    noisy[noisy < 0] = background
    return np.array(noisy, dtype='uint16')
