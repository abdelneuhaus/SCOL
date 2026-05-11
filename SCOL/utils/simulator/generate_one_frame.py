import numpy as np


def add_gaussian_to_frame(frame, amplitude, x0, y0, sigma=1.0, trunc=3):
    height, width = frame.shape

    x_min = max(0, int(np.floor(x0 - trunc * sigma)))
    x_max = min(width, int(np.ceil(x0 + trunc * sigma + 1)))
    y_min = max(0, int(np.floor(y0 - trunc * sigma)))
    y_max = min(height, int(np.ceil(y0 + trunc * sigma + 1)))

    y, x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max))
    gauss = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    frame[x_min:x_max, y_min:y_max] += gauss



def add_gaussian_to_frame_precise(frame, amplitude, x0, y0, sigma=1.0, trunc=5):
    """
    Identique à l'original mais :
    • axes cohérents (x = colonnes, y = lignes)
    • troncature poussée à 5 sigma
    """
    height, width = frame.shape
    x_min = max(0, int(np.floor(x0 - trunc * sigma)))
    x_max = min(width, int(np.ceil(x0 + trunc * sigma + 1)))
    y_min = max(0, int(np.floor(y0 - trunc * sigma)))
    y_max = min(height, int(np.ceil(y0 + trunc * sigma + 1)))

    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max),
                                 np.arange(y_min, y_max),
                                 indexing='xy')

    gauss = amplitude * np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2)
                               / (2 * sigma ** 2))

    frame[y_min:y_max, x_min:x_max] += gauss




def generate_one_frame(molecules, image_size_x, image_size_y, frame=0, sigma=1.0, trunc=4, is_loaded=False):
    img = np.zeros((image_size_x, image_size_y), dtype=np.float16)

    for mol in molecules.values():
        if frame in mol['on_times']:
            x0, y0 = mol['coordinates']
            amp = mol['intensity']
            add_gaussian_to_frame_precise(img, amp, x0, y0, sigma=sigma, trunc=trunc)

    return img, molecules