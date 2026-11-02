import tifffile


def split_raw_image(image_path:str, start_x:int, start_y:int, roi_width:int, roi_height:int, gap:int):
    """
    Split a TIFF/STK image in two separate images.
    
    Args:
        image_path (str): pathway to image
        start_x, start_y (int, int): XY coordinates of the upper right corner ROI.
        roi_width, roi_height (int, int): ROI height and width
        gap (int): size of the gap between the two images in the raw data.
        
    Return:
        2 images
    """
    
    img = tifffile.imread(image_path)

    y1_start = start_y
    y1_end = start_y + roi_height
    x_start = start_x
    x_end = start_x + roi_width
    y2_start = y1_end + gap
    y2_end = y2_start + roi_height

    high_snr = img[..., y1_start:y1_end, x_start:x_end]
    low_snr = img[..., y2_start:y2_end, x_start:x_end]

    return high_snr, low_snr