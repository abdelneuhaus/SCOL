import tifffile
import time as time
import numpy as np

from typing import Union

from noise import add_noise
from generators import generate_one_frame, generate_molecules_data
from io_utils import save_parameters, save_data, load_molecule_data



def SMLM_simulation(frames:np.ndarray, nb_emitters:int, filename:str, randomize:bool=True, 
                    min_intensity:Union[float, int]=500, max_intensity:Union[float, int]=600, ratio:float=0.05, 
                    x_image:int=64, y_image:int=64, length_min:int=1, length_max:int=3, 
                    blink_min:int=1, blink_max:int=3, background_value:Union[float, int]=750, sd_bckg_value:Union[float, int]=6, 
                    is_loaded:bool=False, mask_path:str=None, no_overlap:bool=True, min_distance:int=5):
    """
    Orchestrates the full SMLM (Single Molecule Localization Microscopy) simulation pipeline.

    Args:
        frames (int): Total number of frames to simulate.
        nb_emitters (int): Number of molecules to generate (ignored if is_loaded is True).
        filename (str): Base name for the output .tif and .json files.
        randomize (bool): If True, randomizes the starting frame of the blinking sequences.
        min_intensity (Union[float, int]): Minimum photon count for the emitters.
        max_intensity (Union[float, int]): Maximum photon count for the emitters.
        ratio (float): Scaling factor applied to intensity when loading existing data.
        x_image (int): Image width in pixels.
        y_image (int): Image height in pixels.
        length_min (int): Minimum duration (in frames) of a single ON-state.
        length_max (int): Maximum duration (in frames) of a single ON-state.
        blink_min (int): Minimum number of activation cycles per molecule.
        blink_max (int): Maximum number of activation cycles per molecule.
        background_value (Union[float, int]): Mean value of the background noise (offset).
        sd_bckg_value (Union[float, int]): Standard deviation of the electronic (Gaussian) noise.
        is_loaded (bool): If True, loads molecule data from a file instead of generating it.
        mask_path (str, optional): Path to a TIFF mask for constrained spatial sampling.
        no_overlap (bool): If True, enforces a minimum distance between molecules active in the same frame.
        min_distance (int): The minimum distance (in pixels) required if no_overlap is True.

    Returns:
        None: The function saves the results (.tif, .json, and parameters file).
    """

    if is_loaded:
        points = load_molecule_data()
        nb_emitters = len(points)
        for p in points.values():
            p['intensity'] *= ratio
    else:
        points = generate_molecules_data(frames, nb_emitters, x_image, y_image, randomize, 
                                       min_intensity, max_intensity, length_min, length_max, 
                                       blink_min, blink_max, min_distance, mask_path, no_overlap)

    active_in_frame = {f: [] for f in range(frames)}
    for m_id, m_data in points.items():
        for f in m_data['on_times']:
            if 0 <= f < frames:
                active_in_frame[f].append(m_id)
    full_metadata = []

    with tifffile.TiffWriter(filename) as tif:
        for i in range(frames):
            img, points = generate_one_frame(points, x_image, y_image, frame=i, sigma=1.0, trunc=6)
            for m_id in active_in_frame[i]:
                mol = points[m_id]
                full_metadata.append({
                    'frame': i,
                    'index': m_id,
                    'coordinates': list(mol['coordinates']),
                    'intensity': int(mol['intensity'])
                })

            out = add_noise(img, background_value, sd_bckg_value)
            tif.write(out.astype('uint16'), photometric='minisblack')

    save_data(points, filename)
    save_parameters(filename, frames, nb_emitters, max_intensity, length_min, length_max, 
                    blink_min, blink_max, background_value, sd_bckg_value)



# MAIN
FRAMES = 1000
N_EMITTERS = 500
FILENAME = "high.tif"
RANDOMIZE = True
MIN_INTENSITY, MAX_INTENSITY = 700, 800
RATIO = 0.05, # corresponding to 5% SNR
X_IMAGE, Y_IMAGE = 128, 128
LENGTH_MIN = 2
LENGTH_MAX = 4
BLINK_MIN = 2
BLINK_MAX = 6
BACKGROUND_VALUE = 100
SD_BCKG_VALUE = 25
IS_LOADED = False
MASK_PATH = None
NO_OVERLAP = False
MIN_DISTANCE = 5

start = time.time()
SMLM_simulation(FRAMES, N_EMITTERS, FILENAME, RANDOMIZE, MIN_INTENSITY, MAX_INTENSITY, RATIO, 
                X_IMAGE, Y_IMAGE, LENGTH_MIN, LENGTH_MAX, BLINK_MIN, BLINK_MAX, 
                BACKGROUND_VALUE, SD_BCKG_VALUE, IS_LOADED, MASK_PATH, NO_OVERLAP, MIN_DISTANCE)
end = time.time()
print("Time elapsed:", end - start, "seconds")
