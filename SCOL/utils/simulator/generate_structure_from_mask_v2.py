import json
import tifffile
import time as time
import numpy as np
import tkinter.filedialog as fd

from PIL import Image

from utils import generate_intensity, generate_on_times, add_noise, distance
from saving import save_parameters, save_data
from generate_one_frame import generate_one_frame



def load_3d_mask_coords(path, output_size=(512, 512)):
    """
    
    """
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



def create_molecules_data(
    frames,
    nbr_molecules=20,
    size_x=32,
    size_y=32,
    randomize=True,
    value=100000,
    ii_sd=0,
    off_length_min=1,
    off_length_max=3,
    number_blink_min=1,
    number_blink_max=3,
    min_distance=5,
    mask_path="SMLM_1280_mask.tif",
    no_overlap=True
):
    data = dict()
    mask3d = load_3d_mask_coords(mask_path)

    for i in range(nbr_molecules):
        on_times = generate_on_times(
            frames,
            randomize=randomize,
            off_length_min=off_length_min,
            off_length_max=off_length_max,
            number_blink_min=number_blink_min,
            number_blink_max=number_blink_max
        )

        data[i] = {
            'coordinates': None,
            'intensity': generate_intensity(value, ii_sd),
            'on_times': on_times,
            'shift': 0,
            'model': None
        }

    frame_dict = {frame: [] for frame in range(frames)}

    for mol_id, mol_data in data.items():
        valid = False
        tries = 0

        while not valid:
            tries += 1
            if tries > 5000:
                raise Exception(f"Impossible de placer la molécule {mol_id} après 5000 essais.")

            candidate_coord = generate_emitters_from_coord_list(
                mask3d,
                size_x=size_x,
                size_y=size_y,
                path=mask_path
            )

            valid = True

            if no_overlap:
                for frame in mol_data['on_times']:
                    for existing_coord in frame_dict[frame]:
                        if distance(candidate_coord, existing_coord) < min_distance:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                mol_data['coordinates'] = candidate_coord
                for frame in mol_data['on_times']:
                    frame_dict[frame].append(candidate_coord)

    return data



def generate_stack(
    frames,
    nb_emitters,
    filename,
    randomize=True,
    intensity=1500,
    ii_sd=0,
    x_image=2500,
    y_image=2500,
    length_min=1,
    length_max=3,
    blink_min=1,
    blink_max=3,
    background_value=750,
    sd_bckg_value=6,
    save=True,
    is_loaded=False,
    loaded_data=None,
    mask_path="10px_circle_NPC.tif",
    no_overlap=True
):
    if is_loaded:
        nb_emitters = len(loaded_data)

    points = create_molecules_data(
        frames,
        nbr_molecules=nb_emitters,
        size_x=x_image,
        size_y=y_image,
        randomize=randomize,
        value=intensity,
        ii_sd=ii_sd,
        off_length_min=length_min,
        off_length_max=length_max,
        number_blink_min=blink_min,
        number_blink_max=blink_max,
        mask_path=mask_path,
        no_overlap=no_overlap
    )

    if is_loaded:
        for i in points.keys():
            points[i]['intensity'] = loaded_data[i]['intensity'] * 0.05
            points[i]['model'] = None
            points[i]['coordinates'][0], points[i]['coordinates'][1] = loaded_data[i]['coordinates']
            points[i]['on_times'] = loaded_data[i]['on_times']

    toSave = {}
    with tifffile.TiffWriter(filename) as tif:
        toSave = []
        for i in range(frames):
            data, points = generate_one_frame(points, x_image, y_image, frame=i, sigma=1.0, trunc=6)

            to_save_points = [
                {
                    'frame': int(i),
                    'index': int(u),
                    'coordinates': list(points[u]['coordinates']),
                    'intensity': int(points[u]['intensity'] * 0.05) if is_loaded else int(points[u]['intensity'])
                }
                for u in range(len(points)) if i in points[u]['on_times']
            ]

            toSave += to_save_points

            out = add_noise(data, background_value, sd_bckg_value)
            tif.write(np.array(out, dtype='uint16'), photometric='minisblack')

        save_data(points, filename)
        save_parameters(
            filename,
            frames,
            nb_emitters,
            intensity,
            length_min,
            length_max,
            blink_min,
            blink_max,
            background_value,
            sd_bckg_value
        )



def load_molecule_data():
    filetypes = (('JSON files', '*.json'), ('All files', '*.*'))
    try:
        load_data = fd.askopenfilename(
            title='Open a file',
            initialdir='.',
            filetypes=filetypes
        )

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


# MAIN
INTENSITY = 600
SD_II = 0
RANDOMIZE = True
X_IMAGE, Y_IMAGE = 256, 256
LENGTH_MIN = 2
LENGTH_MAX = 4
BLINK_MIN = 2
BLINK_MAX = 6
BACKGROUND_VALUE = 100
SD_BCKG_VALUE = 25
SAVE = False
IS_LOADED = True
LOADED_DATA =  load_molecule_data()
MASK_PATH = "SMLM_1280_mask.tif"
NO_OVERLAP = False

start = time.time()

generate_stack(
    frames=5000,
    nb_emitters=4500,
    filename="low.tif",
    randomize=RANDOMIZE,
    intensity=1000,  # min intensity
    ii_sd=1100,  # max intensity
    x_image=X_IMAGE,
    y_image=Y_IMAGE,
    length_min=LENGTH_MIN,
    length_max=LENGTH_MAX,
    blink_min=BLINK_MIN,
    blink_max=BLINK_MAX,
    background_value=BACKGROUND_VALUE,
    sd_bckg_value=SD_BCKG_VALUE,
    save=SAVE,
    is_loaded=IS_LOADED,
    loaded_data=LOADED_DATA,
    mask_path=MASK_PATH,
    no_overlap=NO_OVERLAP
)

end = time.time()
print("Time elapsed:", end - start, "seconds")
