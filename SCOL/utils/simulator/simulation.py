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
                    is_loaded:bool=False, mask_path:str="10px_circle_NPC.tif", no_overlap:bool=True, min_distance:int=5):
    """
    
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
RANDOMIZE = True
X_IMAGE, Y_IMAGE = 256, 256
LENGTH_MIN = 2
LENGTH_MAX = 4
BLINK_MIN = 2
BLINK_MAX = 6
BACKGROUND_VALUE = 100
SD_BCKG_VALUE = 25
IS_LOADED = False
MASK_PATH = None
NO_OVERLAP = False

start = time.time()

SMLM_simulation(
    frames=1000,
    nb_emitters=500,
    filename="SMLM_high.tif",
    randomize=RANDOMIZE,
    min_intensity=500,  # min intensity
    max_intensity=550,  # max intensity
    ratio=0.05,
    x_image=X_IMAGE,
    y_image=Y_IMAGE,
    length_min=LENGTH_MIN,
    length_max=LENGTH_MAX,
    blink_min=BLINK_MIN,
    blink_max=BLINK_MAX,
    background_value=BACKGROUND_VALUE,
    sd_bckg_value=SD_BCKG_VALUE,
    is_loaded=IS_LOADED,
    mask_path=MASK_PATH,
    no_overlap=NO_OVERLAP, 
    min_distance=5
)

end = time.time()
print("Time elapsed:", end - start, "seconds")
