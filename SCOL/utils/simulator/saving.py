import os
import json

import numpy as np


def save_parameters(filename:str, frames:int, nb_emitters:int, intensity:float|int, 
                    length_min:int, length_max:int, blink_min: int, blink_max:int, 
                    background_value:float|int, sd_bckg_value:float|int):
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



def save_data(points, filename):
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
    for i, point in enumerate(points):        
        dictionary[i] = {
            'coordinates': point[i]['coordinates'],
            'intensity': int(point[i]['intensity']),
            'on_times': np.array(point[i]['on_times'], dtype='uint16').tolist(),
            'shift': np.array(point[i]['shift'], dtype='uint16').tolist()
        }
    json_object = json.dumps(dictionary, indent = 4)
    with open(filename+".json", "w") as outfile:
        outfile.write(json_object)