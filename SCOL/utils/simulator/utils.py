import tifffile
import numpy as np

from PIL import Image



def add_noise(image_to_noised:np.ndarray, background:float|int, sd:float|int):
    """
    Add simple mixture of Poisson and Gaussian noises to an image containing only PSFs.

    Args:
        image_to_noised (np.ndarray): trained model
        background (float|int): path to low SNR image.
        sd (float|int): path to high SNR image.

    Returns:
        np.ndarray corresponding to the image with pseudocamera noise.
    """
    noisy = np.random.poisson(image_to_noised).astype(np.float64)
    noisy += np.random.normal(loc=background, scale=sd, size=image_to_noised.shape)
    return np.maximum(noisy, background).astype(np.uint16)



def distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (ArrayLike): Coordinates of the first point (list, tuple, or np.array).
        p2 (ArrayLike): Coordinates of the second point (list, tuple, or np.array).

    Returns:
        float: The Euclidean distance (L2 norm) between p1 and p2.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))



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