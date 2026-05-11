"""
Our pipeline is:
    - dimension: temporal (time = batch if artefacts)
    - Fixed pattern suppression
    - Variance stabilization
    - Noise2Self-CB: +spatial features +balance training data
"""
import time
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR
from aydin.features.standard_features import StandardFeatureGenerator
from aydin.it.fgr import ImageTranslatorFGR
from aydin.it.transforms.fixedpattern import FixedPatternTransform
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.variance_stabilisation import VarianceStabilisationTransform
from aydin.it.transforms.range import RangeTransform
from aydin.regression.cb import CBRegressor



# Advanced Noise2Self with preprocessing and CatBoost network
noisy_image = tiff.imread("data/SIMULATION/Training/Low.tif")
print("Image shape:", noisy_image.shape)

# Transforms (dans l’ordre du JSON)
transforms = [
    {"class": FixedPatternTransform, "kwargs": {"axes": None, "do_postprocess": False, "percentile": 1.0, "priority": 0.09, "sigma": 0.5}},
    {"class": PaddingTransform, "kwargs": {"do_postprocess": True, "min_length_to_pad": 8, "mode": "reflect", "pad_width": 3, "priority": 0.9}},
    {"class": VarianceStabilisationTransform, "kwargs": {"do_postprocess": True, "leave_as_float": True, "mode": "anscomb", "priority": 0.11}},
    {"class": RangeTransform, "kwargs": {"clip": True, "do_postprocess": True, "force_float_datatype": False, "mode": "minmax", "percentile": None, "priority": 0.2}},
]

# Instancier Noise2SelfFGR
n2s = Noise2SelfFGR(
    variant="cb",
    it_transforms=transforms,
    lower_level_args={
        "feature_generator": {
            "class": StandardFeatureGenerator,
            "kwargs": {
                "dct_max_freq": 0.5,
                "decimate_large_scale_features": True,
                "extend_large_scale_features": False,
                "include_corner_features": False,
                "include_dct_features": False,
                "include_fine_features": True,
                "include_line_features": False,
                "include_lowpass_features": True,
                "include_median_features": False,
                "include_random_conv_features": False,
                "include_scale_one": False,
                "include_spatial_features": True,
                "max_level": 13,
                "min_level": 0,
                "num_lowpass_features": 8,
                "num_sinusoidal_features": 0,
                "scale_one_width": 3,
                "spatial_features_coarsening": 2,
            },
        },
        "it": {
            "class": ImageTranslatorFGR,
            "kwargs": {
                "balance_training_data": True,
                "blind_spots": None,
                "favour_bright_pixels": 0.0,
                "max_memory_usage_ratio": 0.9,
                "max_tiling_overhead": 0.1,
                "max_voxels_for_training": None,
                "tile_max_margin": None,
                "tile_min_margin": 8,
                "voxel_keep_ratio": 1.0,
            },
        },
        "regressor": {
            "class": CBRegressor,
            "kwargs": {
                "compute_load": 0.95,
                "gpu": True,
                "gpu_devices": None,
                "gpu_use_pinned_ram": None,
                "learning_rate": None,
                "loss": "l1",
                "max_bin": None,
                "max_num_estimators": None,
                "min_num_estimators": None,
                "num_leaves": None,
                "patience": 32,
            },
        },
    },
)

# Entraînement
debut = time.time()
n2s.train(noisy_image)

# Denoising
denoised_image = n2s.denoise(noisy_image)
end = time.time()

print(f"Restoration took {(end-debut)/60:.2f} minutes")

# Affichage et sauvegarde
plt.imshow(denoised_image[400], cmap="gray")
plt.show()

tiff.imwrite("data/SIMULATION/Prediction/n2s.tif", np.array(denoised_image, dtype="uint16"))
