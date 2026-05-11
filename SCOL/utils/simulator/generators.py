import random
import tifffile
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



def add_gaussian_to_frame_precise(frame, amplitude, x0, y0, sigma=1.0, trunc=5):
    """
    Add a 2D Gaussian spot to an existing image frame with sub-pixel precision.

    This function calculates a Gaussian distribution over a local bounding box 
    defined by the truncation radius and adds it to the input frame.

    Args:
        frame (np.ndarray): The 2D image array to which the Gaussian will be added.
        amplitude (float): The peak intensity (height) of the Gaussian spot.
        x0 (float): The sub-pixel X-coordinate of the spot center.
        y0 (float): The sub-pixel Y-coordinate of the spot center.
        sigma (float): Standard deviation of the Gaussian (controls width). Default to 1.0.
        trunc (int): Number of sigmas used to define the calculation window size. Default to 5.

    Returns:
        None: The function modifies the 'frame' argument in-place.
    """
    height, width = frame.shape
    x_min = max(0, int(np.floor(x0 - trunc * sigma)))
    x_max = min(width, int(np.ceil(x0 + trunc * sigma + 1)))
    y_min = max(0, int(np.floor(y0 - trunc * sigma)))
    y_max = min(height, int(np.ceil(y0 + trunc * sigma + 1)))

    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max),
                                 np.arange(y_min, y_max),
                                 indexing='xy')

    gauss = amplitude * np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
    frame[y_min:y_max, x_min:x_max] += gauss




def generate_one_frame(molecules, image_size_x, image_size_y, frame=0, sigma=1.0, trunc=5):
    """
    Generate a single simulation frame by rendering active emitters as Gaussian spots.

    Args:
        molecules (dict): A dictionary where keys are emitter IDs and values are 
            dictionaries containing 'coordinates', 'intensity', and 'on_times'.
        image_size_x (int): Number of pixels along the X-axis (height).
        image_size_y (int): Number of pixels along the Y-axis (width).
        frame (int): The current frame index to simulate. Defaults to 0.
        sigma (float): Standard deviation of the Gaussian PSF in pixels. Defaults to 1.0.
        trunc (int): The number of sigmas at which the Gaussian is truncated. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - img (np.ndarray): The generated frame as a float16 2D array.
            - molecules (dict): The original molecules dictionary.
    """
    img = np.zeros((image_size_x, image_size_y), dtype=np.float16)

    for mol in molecules.values():
        if frame in mol['on_times']:
            x0, y0 = mol['coordinates']
            amp = mol['intensity']
            add_gaussian_to_frame_precise(img, amp, x0, y0, sigma=sigma, trunc=trunc)

    return img, molecules




def generate_emitters_from_coord_list(coord_list, size_x, size_y, path, rng=None):
    """
    Generate a single emitter coordinates by sampling and rescaling from a reference stack. 

    Args:
        coord_list (list[tuple]): List of (z, y, x) coordinates to sample from.
        size_x (int): Target width of the simulation grid.
        size_y (int): Target height of the simulation grid.
        path (str): Path to the reference .tif stack to determine original dimensions.
        rng (np.random.Generator, optional): NumPy random number generator for reproducibility. 

    Returns:
        list[float]: A list containing the rescaled [x, y] coordinates.
    """
    stack = tifffile.imread(path)  # (Z,H,W)
    original_y = stack.shape[0]
    original_x = stack.shape[1]
    factor_x = original_x / size_x
    factor_y = original_y / size_y
    if rng is None:
        rng = np.random.default_rng()
    index = rng.integers(0, len(coord_list))
    z, y, x = coord_list[index]
    x_f = x + rng.uniform(0, 1)
    y_f = y + rng.uniform(0, 1)

    return [float(x_f / factor_x), float(y_f / factor_y)]