import numpy as np


# def add_gaussian_to_frame(frame, amplitude, x0, y0, sigma=1.0, trunc=3):
#     height, width = frame.shape

#     x_min = max(0, int(np.floor(x0 - trunc * sigma)))
#     x_max = min(width, int(np.ceil(x0 + trunc * sigma + 1)))
#     y_min = max(0, int(np.floor(y0 - trunc * sigma)))
#     y_max = min(height, int(np.ceil(y0 + trunc * sigma + 1)))

#     y, x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max))
#     gauss = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

#     frame[x_min:x_max, y_min:y_max] += gauss



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