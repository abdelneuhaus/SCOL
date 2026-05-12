import os
import json
import tifffile

import numpy as np
import tkinter.filedialog as fd

from PIL import Image
from typing import Union


def save_parameters(filename:str, frames:int, nb_emitters:int, intensity:Union[float, int], 
                    length_min:int, length_max:int, blink_min: int, blink_max:int, 
                    background_value:Union[float, int], sd_bckg_value:Union[float, int]):
    """
    Save simulation parameters for the corresponding generated image.

    Args:
        filename (str): The name or path of the generated image file.
        frames (int): Total number of frames in the simulation.
        nb_emitters (int): Number of simulated molecules/emitters.
        intensity (float|int): Integrated intensity value used for emitters.
        length_min (int): Minimum ON-time duration (in frames).
        length_max (int): Maximum ON-time duration (in frames).
        blink_min (int): Minimum number of blinks per molecule.
        blink_max (int): Maximum number of blinks per molecule.
        background_value (float|int): Mean background intensity value.
        sd_bckg_value (float|int): Standard deviation of the background noise.

    Returns:
        None: This function writes a file and does not return a value.
    """
    
    base_name = os.path.splitext(filename)[0]
    output_path = f"{base_name}_parameters.txt"
    to_save = (
        f"Number of frames: {frames}\n"
        f"Numbers of emitters: {nb_emitters}\n"
        f"Integrated Intensity: {intensity}\n"
        f"Background value: {background_value}, sd: {sd_bckg_value}\n"
        f"ON duration: [{length_min}, {length_max}]\n"
        f"Number of blinks: [{blink_min}, {blink_max}]"
    )
    
    with open(output_path, 'w') as f:
        f.write(to_save)



def save_data(points, filename:str):
    """
    Save simulation data and emitter properties to a JSON file.

    Args:
        points (list[dict]): A list of dictionaries, where each dictionary contains 
            the properties of an emitter (coordinates, intensity, on_times, shift).
        filename (str): The name or path of the file (extension .tif is handled).

    Returns:
        None: This function writes a JSON file and does not return a value.
    """
    filename = filename.replace('.tif','')
    dictionary = dict()
    for mol_id, mol_data in points.items():        
        dictionary[mol_id] = {
            'coordinates': mol_data['coordinates'],
            'intensity': int(mol_data['intensity']),
            'on_times': np.array(mol_data['on_times'], dtype='uint16').tolist(),
            'shift': mol_data['shift']
        }
    
    json_object = json.dumps(dictionary, indent=4)
    with open(filename + ".json", "w") as outfile:
        outfile.write(json_object)



def load_molecule_data():
    """
    Open a file dialog to load and preprocess molecule data from a JSON file.

    Returns:
        dict: A dictionary of molecule data where keys are integer IDs
    """

    filetypes = (('JSON files', '*.json'), ('All files', '*.*'))
    try:
        load_data = fd.askopenfilename(title='Open a file', initialdir='.', filetypes=filetypes)
        with open(load_data, 'r') as f:
            data_loaded = json.load(f)
        data_loaded = {int(k): v for k, v in data_loaded.items()}
        if '_diffusion' in load_data:
            for i in range(len(data_loaded)):
                data_loaded[i]['on_times'] = [data_loaded[i]['frame']]
                data_loaded[i]['shift'] = 0
        print("Done")
        return data_loaded
    except:
        print("No JSON loaded")


def load_3d_mask_coords(path, output_size=(512, 512)):
    """
    Extract spatial coordinates from a binary mask after resizing.

    Args:
        path (str): Path to the input .tif file.
        output_size (tuple[int, int]): Target (width, height) for resizing the mask. 

    Returns:
        list[tuple[int, int, int]]: A list of (z, y, x) coordinates where the mask is active. 
        Note: z is currently hardcoded to 0, but can be changed to read 3D mask
    """
    if path is None:
        return None
    
    stack = tifffile.imread(path)  # shape (Z,H,W)
    coords = []
    img = Image.fromarray(stack)
    img = img.convert("L")
    img_resized = img.resize(output_size, resample=Image.Resampling.BILINEAR)
    arr = np.array(img_resized)
    binary_mask = (arr > (arr.max() * 0.5)).astype(np.uint8)
    ys, xs = np.where(binary_mask == 1)
    coords.extend([(0, y, x) for y, x in zip(ys, xs)])
    return coords