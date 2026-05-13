import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


def get_last_file(path, name: str, sort_mode: Literal["time", "alpha"] = "alpha") -> str:
	"""
	Retrieves the last file in a directory that contains a specific substring in its name.
    
    Args:
        path (Union[str, Path]): The directory path to search in.
        name (str): The substring to look for within the filenames.
        sort_mode (Literal["time", "alpha"], optional): The sorting method to apply.

    Returns:
        str: The full path of the last matching file as a string.
    """
    
	try:
		folder = Path(path)
		if not folder.is_dir(): return "" 
		files = [p for p in folder.iterdir() if p.is_file() and name in p.name]
		if not files: return ""
		if sort_mode == "time": files.sort(key=lambda p: p.stat().st_mtime)
		else: files.sort(key=lambda p: p.name)
		return str(files[-1])
	except Exception as e:
		print(f"Error while searching for the file: {e}")
		return ""
    


def read_coefficients_from_file(file_path: str):
    """
    Read and extract X and Y coefficents from a specific text file.

    Args:
        file_path (str): File path to coefficients file.

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple with coeff_x and coeff_y.
    """

    with open(file_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("Le fichier doit contenir au moins trois lignes.")
    coeff_x = np.array(list(map(float, lines[1].split())), dtype=np.float64)
    coeff_y = np.array(list(map(float, lines[2].split())), dtype=np.float64)
    return coeff_x, coeff_y



def calculate_new_coordinates(x, y, cfx, cfy):
    """
    Reads and extracts X and Y coefficients from a text file.

    Args:
        file_path (str): The path to the text file containing the coefficients.

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple with coeff_x and coeff_y.
    """

    x2, y2 = x * x, y * y
    x3, y3 = x2 * x, y2 * y

    new_x = (
        cfx[0] * x3 + cfx[1] * y3 + cfx[2] * x2 * y + cfx[3] * x * y2 +
        cfx[4] * x2 + cfx[5] * y2 + cfx[6] * x * y + cfx[7] * x +
        cfx[8] * y + cfx[9]
    )
    new_y = (
        cfy[0] * x3 + cfy[1] * y3 + cfy[2] * x2 * y + cfy[3] * x * y2 +
        cfy[4] * x2 + cfy[5] * y2 + cfy[6] * x * y + cfy[7] * x +
        cfy[8] * y + cfy[9]
    )
    return new_x, new_y



def ground_truth_coordinates(df):
    """
    Parses and flattens ground truth coordinates from a raw DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data.

    Returns:
        pd.DataFrame: A formatted DataFrame with columns ["Plane", "X", "Y"].
    """

    coords = []
    for i in range(len(df)):
        xy = df['approximative coordinates (x,y) '][i].replace('(', '').replace(')', '').split(',')
        frames = df['blinking frames '][i].replace('[', '').replace(']', '').replace("'", '').split(',')
        for f in set(frames):
            f = f.strip()
            if f.isdigit() and int(f) != 31:
                coords.append((int(f), float(xy[0]), float(xy[1])))
    return pd.DataFrame(coords, columns=["Plane", "X", "Y"])



def create_coordinates_from_pt(df):
    """
    Cleans and standardizes a DataFrame containing coordinate data.

    Args:
        df (pd.DataFrame): The input DataFrame and must contain columns named 'Plane', 'X', and 'Y'.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing only the rounded 'Plane', 'X', and 'Y' columns, with a reset index.
    """

    df = df[['Plane', 'X', 'Y']].round(8)
    df.columns = ['Plane', 'X', 'Y']
    return df.reset_index(drop=True)



def compare_coordinates_with_sigma(df_gt, df_pred, cfx=None, cfy=None, sigma=1.0, pixel_size=160):
    """
    Evaluates predicted molecule coordinates against ground truth using a distance threshold.

    Args:
        df_gt (pd.DataFrame): The ground truth DataFrame. Must contain 'Plane', 'X', and 'Y' columns.
        df_pred (pd.DataFrame): The predicted DataFrame. Must contain 'Plane', 'X', and 'Y' columns.
        cfx (Union[list, np.ndarray], optional): 10 coefficients for X-axis polynomial transformation. 
        cfy (Union[list, np.ndarray], optional): 10 coefficients for Y-axis polynomial transformation. 
        sigma (float, optional): The maximum allowable Euclidean distance (in pixels) for a prediction to be considered a valid match.
        pixel_size (float, optional): The physical size of a pixel in nanometers, used to convert the final mean distance error.

    Returns:
        None: This function does not return a value. It prints the calculated metrics directly to the standard output.
    """

    tp = 0
    distances = []
    
    if cfx is not None and cfy is not None:
        df_pred = df_pred.copy()
        new_x, new_y = calculate_new_coordinates(df_pred["X"].values, df_pred["Y"].values, cfx, cfy)
        df_pred["X"] = new_x
        df_pred["Y"] = new_y
    planes = sorted(set(df_gt["Plane"]).intersection(set(df_pred["Plane"])))

    for p in planes:
        gt = df_gt[df_gt["Plane"] == p][["X", "Y"]].to_numpy()
        pred = df_pred[df_pred["Plane"] == p][["X", "Y"]].to_numpy()

        if len(gt) == 0 or len(pred) == 0:
            continue

        dists = np.sqrt(((gt[:, np.newaxis, :] - pred[np.newaxis, :, :]) ** 2).sum(axis=2))
        matches = []
        for i in range(dists.shape[0]):
            for j in range(dists.shape[1]):
                if dists[i, j] <= sigma:
                    matches.append((i, j, dists[i, j]))

        matches.sort(key=lambda x: x[2])
        used_gt = set()
        used_pred = set()

        for i, j, dist in matches:
            if i not in used_gt and j not in used_pred:
                used_gt.add(i)
                used_pred.add(j)
                tp += 1
                distances.append(dist)

    fp = len(df_pred) - tp
    fn = len(df_gt) - tp

    print("TP (%):", tp / (tp + fn) * 100 if (tp + fn) else 0)
    print("FP (%):", fp / (tp + fp) * 100 if (tp + fp) else 0)
    print("FN (%):", fn / (tp + fn) * 100 if (tp + fn) else 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    if distances:
        distances_nm = np.array(distances) * pixel_size
        print("Mean distance error if TP (nm):", np.mean(distances_nm))
    else:
        print("Aucune correspondance trouvée.")



#### MAIN ####
# # Loading the data
DATA1 = get_last_file("C:/Git/SCOL/data/SIMULATION/Prediction/target_PALM_Tracer", "localizations")
DATA2 = get_last_file("C:/Git/SCOL/data/SIMULATION/Prediction/noisy_CARE_PALM_Tracer", "localizations")
GT = pd.read_csv(DATA1, sep=',', engine='python')
OUTPUT = pd.read_csv(DATA2, sep=',', engine='python')

# Coefficients
# COEFF_PATH  = "C:/Git/SCOL/data/GROUND_TRUTH/SPT/DUALVIEW.PT/DUALVIEW_2CFit.txt"
# coeff_x, coeff_y = read_coefficients_from_file(COEFF_PATH)
    
# Extracting coordinates
output = create_coordinates_from_pt(OUTPUT)
gt = create_coordinates_from_pt(GT)
compare_coordinates_with_sigma(gt, output, cfx=None, cfy=None, sigma=1.0, pixel_size=160)