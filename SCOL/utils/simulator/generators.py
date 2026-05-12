import random
import tifffile
import numpy as np

from typing import Union

from io_utils import load_3d_mask_coords


def generate_intensity(high:Union[float, int]=500, low:Union[float, int]=800):
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

    radius = int(trunc * sigma) # patch radius
    x_min_th, x_max_th = int(x0) - radius, int(x0) + radius + 1
    y_min_th, y_max_th = int(y0) - radius, int(y0) + radius + 1
    x_min = max(0, x_min_th)
    x_max = min(frame.shape[1], x_max_th)
    y_min = max(0, y_min_th)
    y_max = min(frame.shape[0], y_max_th)

    if x_min >= x_max or y_min >= y_max:
        return 
    
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    gauss = amplitude * np.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (2 * sigma**2))
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
    if rng is None:
        rng = np.random.default_rng()

    if coord_list is None:
        x_f = rng.uniform(0, size_x)
        y_f = rng.uniform(0, size_y)
        return [float(x_f), float(y_f)]
    
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




def generate_molecules_data(frames, nbr_molecules:int, size_x:int=64, size_y:int=64, randomize:bool=True, 
                          min_intensity:Union[float, int]=500, max_intensity:Union[float, int]=600, 
                          off_length_min:int=1, off_length_max:int=3, number_blink_min:int=1, number_blink_max:int=3, 
                          min_distance:int=5, mask_path:str=None, no_overlap:bool=True, 
                          diff_coeffs:Union[list, np.ndarray]=[0.0], proportions:Union[list, np.ndarray]=[1.0]):
    """
    Generate a full dataset of emitters with spatial and temporal constraints.

    Args:
        frames (int): Total number of frames in the simulation.
        nbr_molecules (int): Number of molecules to simulate.
        size_x (int): Target simulation width. Defaults to 64.
        size_y (int): Target simulation height. Defaults to 64.
        randomize (bool): Whether to randomize blinking events. Defaults to True.
        min_intensity (float): Minimum intensity value. Defaults to 500.
        max_intensity (float): Maximum intensity value. Defaults to 600.
        off_length_min (int): Minimum ON-time duration. Defaults to 1.
        off_length_max (int): Maximum ON-time duration. Defaults to 3.
        number_blink_min (int): Minimum number of blinks per molecule. Defaults to 1.
        number_blink_max (int): Maximum number of blinks per molecule. Defaults to 3.
        min_distance (int): Minimum Euclidean distance required between two emitters active in the same frame. Defaults to 5.
        mask_path (str): Path to the reference mask TIFF file. 
        no_overlap (bool): If True, enforces the min_distance constraint.
        diff_coeffs (Union[list, np.ndarray]): List of diffusion coefficient(s). Can handle populations.
        proportions (Union[list, np.ndarray]): List of proportions of each population. Sum must equals 1. Should be same length as diff_coeffs.

    Returns:
        dict: A dictionary where keys are molecule IDs (int) and values individual molecules metadata.

    Raises:
        Exception: If a molecule cannot be placed after 1000 attempts due to overlap constraint.
    """

    data = dict()
    min_dist_sq = min_distance ** 2
    mask3d = load_3d_mask_coords(mask_path)
    frame_dict = {f: [] for f in range(frames)}

    rng = np.random.default_rng()
    assigned_D = rng.choice(diff_coeffs, size=nbr_molecules, p=proportions)

    for i in range(nbr_molecules):
        on_times = generate_on_times(frames, randomize, off_length_min, off_length_max, number_blink_min, number_blink_max)
        data[i] = {
            'coordinates': None,
            'intensity': generate_intensity(min_intensity, max_intensity),
            'on_times': on_times,
            'shift': 0,
            'model': None,
            'D': float(assigned_D[i])
        }

    for mol_id, mol_data in data.items():
        valid = False
        tries = 0
        on_frames = mol_data['on_times']

        while not valid:
            tries += 1
            if tries > 1000:
                raise Exception(f"Saturation : Impossible de placer la molécule {mol_id}")

            candidate_coord = generate_emitters_from_coord_list(mask3d, size_x, size_y, mask_path)
            valid = True
            if no_overlap:
                for frame in on_frames:
                    coords = frame_dict[frame]
                    if not coords:
                        continue
                    existing_arr = np.array(coords)
                    diff = existing_arr - candidate_coord
                    dist_sq = np.sum(diff**2, axis=1)
                    if np.any(dist_sq < min_dist_sq):
                        valid = False
                        break
            if valid:
                mol_data['coordinates'] = candidate_coord
                for frame in on_frames:
                    frame_dict[frame].append(candidate_coord)

    return data