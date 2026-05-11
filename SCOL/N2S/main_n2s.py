"""
Our pipeline is:
    - dimension: temporal (time = batch if artefacts)
    - Fixed pattern suppression
    - Variance stabilization
    - Noise2Self-CB: +spatial features +balance training data
"""

import time
import tifffile as tiff
import matplotlib.pyplot as plt

from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


# Basic Noise2Self using the “Feature Generation & Regression” approach-based restoration
noisy_image = tiff.imread("data/SIMULATION/Training/Low.tif")
print("Image shape:", noisy_image.shape)
debut = time.time()
n2s = Noise2SelfFGR()
n2s.train(noisy_image)
denoised_image = n2s.denoise(noisy_image)
end = time.time()
print(f"Restoration took {(end-debut)/60} minutes")

plt.imshow(denoised_image[0], cmap="gray")
plt.show()
tiff.imwrite("data/SIMULATION/Prediction/n2s.tif", denoised_image.astype("float16"))
