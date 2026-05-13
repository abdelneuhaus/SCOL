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
                    is_loaded:bool=False, mask_path:str=None, no_overlap:bool=True, min_distance:int=5,
                    diff_coeffs:Union[list, np.ndarray]=[0.0], proportions:Union[list, np.ndarray]=[1.0], frame_time:float=0.03, pixel_size:float=0.1):
    """
    Orchestrates the full SMLM simulation pipeline.

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
        diff_coeffs (Union[list, np.ndarray]): List of diffusion coefficient(s). Can handle populations.
        proportions (Union[list, np.ndarray]): List of proportions of each population. Sum must equals 1. Should be same size as diff_coeffs.
        frame_time (float): Pseudo frame rate in ms to compute brownian displacement.
        pixel_size (float): Pseudo pixel size in μm to compute brownian displacement.

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
                                       blink_min, blink_max, min_distance, mask_path, no_overlap,
                                       diff_coeffs, proportions)
        for mol in points.values():
            mol['trajectory'] = {0: list(mol['coordinates'])}

    active_in_frame = {f: [] for f in range(frames)}
    for m_id, m_data in points.items():
        for f in m_data['on_times']:
            if 0 <= f < frames:
                active_in_frame[f].append(m_id)
                
    full_metadata = []
    rng = np.random.default_rng()
    image_stack = [] 
    for i in range(frames):
        if is_loaded:
            for m_id, mol in points.items():
                if str(i) in mol['trajectory']: 
                    mol['coordinates'] = mol['trajectory'][str(i)]

        img, points = generate_one_frame(points, x_image, y_image, frame=i, sigma=1.0, trunc=6)        
        for m_id in active_in_frame[i]:
            mol = points[m_id]
            full_metadata.append({
                'frame': i,
                'index': m_id,
                'coordinates': list(mol['coordinates']),
                'intensity': int(mol['intensity'])
            })

        if not is_loaded:
            for m_id, mol in points.items():
                if mol.get('D', 0.0) > 0:
                    sigma_jump = np.sqrt(2 * mol['D'] * frame_time) / pixel_size
                    mol['coordinates'][0] += rng.normal(0, sigma_jump)
                    mol['coordinates'][1] += rng.normal(0, sigma_jump)
                mol['trajectory'][i+1] = list(mol['coordinates'])

        out = add_noise(img, background_value, sd_bckg_value)
        image_stack.append(out.astype('uint16')) 
    stack_array = np.array(image_stack) 
    
    tifffile.imwrite(filename, stack_array, imagej=True, photometric='minisblack')
    save_data(points, filename)
    save_parameters(filename, frames, nb_emitters, max_intensity, length_min, length_max, 
                    blink_min, blink_max, background_value, sd_bckg_value)



# MAIN
FRAMES = 1000
N_EMITTERS = 50
FILENAME = "high.tif"
RANDOMIZE = True
MIN_INTENSITY, MAX_INTENSITY = 1000, 1100
RATIO = 0.05 # corresponding to 5% SNR, ratio of 1.0 means same SNR
X_IMAGE, Y_IMAGE = 64, 64
LENGTH_MIN = 4
LENGTH_MAX = 40
BLINK_MIN = 2
BLINK_MAX = 5
BACKGROUND_VALUE = 100
SD_BCKG_VALUE = 25
IS_LOADED = False
MASK_PATH = None
NO_OVERLAP = False
MIN_DISTANCE = 5
DIFF_COEFF=[0.05, 0.02, 0.0]
PROPORTIONS=[0.34, 0.33, 0.33]
FRAME_TIME=0.02
PIXEL_SIZE=0.16

start = time.time()
SMLM_simulation(FRAMES, N_EMITTERS, FILENAME, RANDOMIZE, MIN_INTENSITY, MAX_INTENSITY, RATIO, 
                X_IMAGE, Y_IMAGE, LENGTH_MIN, LENGTH_MAX, BLINK_MIN, BLINK_MAX, 
                BACKGROUND_VALUE, SD_BCKG_VALUE, IS_LOADED, MASK_PATH, NO_OVERLAP, MIN_DISTANCE,
                DIFF_COEFF, PROPORTIONS, FRAME_TIME, PIXEL_SIZE)
end = time.time()
print("Time elapsed:", end - start, "seconds")
