import random
import numpy as np


def generate_intensity(high:float|int=500, low:float|int=800):
    """
    Generate an int from a range.

    Args:
        high (float|int): lower bound of the intensity range
        low (float|int): higher bound of the intensity range

    Returns:
        int corresponding to intensity value.
    """
    return int(np.random.randint(high, low))




def generate_on_times(frames, randomize=True, off_length_min=1, off_length_max=3, number_blink_min=1, number_blink_max=3):
    """
    Generate a list of frame indices where an emitter is active (ON state).

    Args:
        frames (int): Total number of frames in the simulation.
        randomize (bool): If True, generates random blinking events throughout the sequence.
            If False, returns a continuous block of the last 11 frames. Defaults to True.
        off_length_min (int): Minimum duration of a single ON event (in frames).
        off_length_max (int): Maximum duration of a single ON event (in frames).
        number_blink_min (int): Minimum number of blinking events to generate.
        number_blink_max (int): Maximum number of blinking events to generate.

    Returns:
        list[int]: A sorted list of unique frame indices where the emitter is ON.
    """

    if not randomize:
        return list(range(max(0, frames - 10), frames + 1))
    
    blink_set = set()
    number_blink = random.randint(number_blink_min, number_blink_max)
    for _ in range(number_blink):
        length = random.randint(off_length_min, off_length_max)
        start = random.randint(0, max(0, frames - length))
        blink_set.update(range(start, start + length))
    return sorted(blink_set)
    


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
    noisy = np.random.poisson(image_to_noised).astype(np.float64)
    noisy += np.random.normal(loc=background, scale=sd, size=image_to_noised.shape)
    return np.maximum(noisy, background).astype(np.uint16)



def distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (ArrayLike): Coordinates of the first point (list, tuple, or np.array).
        p2 (ArrayLike): Coordinates of the second point (list, tuple, or np.array).

    Returns:
        float: The Euclidean distance (L2 norm) between p1 and p2.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))