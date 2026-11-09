# SCOL — CARE denoising for single-molecule imaging

Tools to train and apply [CARE](https://github.com/CSBDeep/CSBDeep) (CSBDeep) to single-molecule data
(single-particle tracking and SMLM), with a self-supervised baseline (Noise2Self, through
[Aydin](https://github.com/royerlab/aydin)) and PALMTracer for detection and localization.

The training pairs come from a **dual-view acquisition**: one camera image is split into a high-SNR
channel (**HIGH**, the training target) and a low-SNR channel (**LOW**, the network input). Both
channels see the same scene at the same time, with a different number of photons.

## Repository layout

```
SCOL/
├── processing/                     preparation of the training data (kernel "Python (Main)")
│   ├── split_raw_image.py          split a dual-view movie into HIGH and LOW
│   ├── subpixel_transformation.py  register the HIGH onto the LOW grid (polynomial from PALMTracer 2CFit)
│   ├── create_paired_dataset.py    paired (HIGH, LOW) patches centered on molecules (PALMTracer detection)
│   ├── create_paired_dataset_mask.py   same patches + localization masks (experimental L5 loss)
│   ├── measure_residual_shift.py   residual HIGH/LOW misalignment after registration
│   ├── Parsing.py, utils.py        PALMTracer outputs and helpers
│   └── DLL/                        PALMTracer libraries (.dylib for macOS, .dll for Windows)
├── CARE/                           training and prediction (kernel "Python (CSBDeep)")
│   ├── data_processing.py          patches -> training and validation .npz
│   ├── data_augmentation.py        rotations and flips (x8), training set only
│   ├── generate_config.py          training configuration
│   ├── training.py                 CARE training (MAE loss), run_info.json in the model folder
│   ├── l5.py                       experimental L5 loss (MAE weighted by localization masks)
│   ├── prediction.py               denoising of a movie with a trained model
│   └── check_training.py           visual check of a training run
├── notebooks/
│   ├── preprocessing.ipynb         split, registration, training patches
│   ├── CARE_training.ipynb         .npz, augmentation, training
│   ├── CARE_prediction.ipynb       denoising with a trained model
│   ├── N2S_denoising.ipynb         Noise2Self baseline (kernel "Python (Aydin)")
│   ├── test_L5_dataset.ipynb       patches + masks for the L5 loss
│   └── test_L5_training.ipynb      training with the L5 loss
└── utils/
    ├── simulator/                  simulated single-molecule movies
    ├── metrics.py                  coordinate comparison on simulated data
    └── check_gpu.py
```

## Installation

macOS (Apple silicon), Python 3.9:

```bash
bash install.sh        # Windows: install.bat
```

This creates three environments, each registered as a Jupyter kernel:

| Environment | Kernel | Used for |
|---|---|---|
| `venv_csbdeep` | Python (CSBDeep) | CARE training and prediction (CSBDeep 0.8.1, TensorFlow 2.10 + Metal) |
| `venv_aydin` | Python (Aydin) | Noise2Self baseline |
| `venv` | Python (Main) | preprocessing, patches, PALMTracer |

## Workflow

### 1. Preprocessing (`preprocessing.ipynb`)
1. **Split** the raw dual-view movie into `<name>_high_snr.tif` and `<name>_low_snr.tif`
   (same ROI size for both, one `x0, y0` per ROI).
2. **Register the HIGH onto the LOW grid** with the PALMTracer 2CFit coefficients (backward mapping,
   linear interpolation). Pixels outside the field are filled with the background median and a
   `_validmask.tif` is saved.
   - Always register the HIGH, never the LOW: the LOW is the network input, and interpolating it
     changes its noise. Prediction is done on raw LOW movies.
   - The fit must go from the output grid to the source (LOW → HIGH) for backward mapping.
3. **Check the registration** with `measure_residual_shift.py` (paths at the top of the file):
   median offset close to 0 and no gradient across the field. A residual offset is learned by the
   network and shifts all its localizations.
4. **Build the training patches** (`build_paired_roi_stacks_batch`): one HIGH frame and
   `t_window_low` LOW frames (1 or 3) per patch, 75 % of the patches centered on a molecule
   detected on the HIGH, fixed seed. Output: `Training/High`, `Training/Low` and `dataset_info.json`.

Detection threshold: about 3.5 × the noise of the second wavelet plane (MAD), measured on each movie.

### 2. Training (`CARE_training.ipynb`)
1. `do_data_processing`: training and validation `.npz` (random split of 5 to 15 % of the patches,
   done before augmentation).
2. `apply_data_augmentation_to_npz` (optional): rotations and flips, training set only.
3. `do_training`: N-to-1 CARE (one output frame), MAE loss, `unet_residual = False`. Each model folder
   contains `run_info.json` (git commit, data, seed, parameters, best epoch).

### 3. Prediction (`CARE_prediction.ipynb`)
`do_denoising_3D` on raw LOW movies.

### 4. Noise2Self baseline (`N2S_denoising.ipynb`)
Self-supervised denoising of the LOW movie itself, with no HIGH and no registration.

### 5. Evaluation
Localize every output (HIGH registered, LOW, CARE, N2S) with PALMTracer, with the same settings
for all, on the same frames. Compare to the registered HIGH:
- one-to-one matching per frame, Jaccard index at 160 and 240 nm;
- **at equal number of detections** (keep the N brightest of each output), so that the comparison
  does not depend on the detection threshold;
- for tracking data, frame-to-frame displacements and MSD of the same molecules.

Evaluate on a movie that was not used for training.

## Experimental: L5 loss (`CARE/l5.py`)
MAE weighted by the PALMTracer localizations of the HIGH (mask = second channel of Y, read by the
loss only; the model and the prediction are unchanged):

```
W  = 1 + w · (target squares ∪ predicted squares) + λ · (predicted squares \ target squares)
L5 = mean(W · |prediction − HIGH|)
```

`w = λ = 0` gives the standard MAE. On tracking data, `w = 10, λ = 10` gave about +0.01 in Jaccard
at equal number of detections, mostly a threshold effect. Not used by default.

## Practical notes
- **Temporal window.** With `t_window_low = 3` the network uses frames t − 1 and t + 1: much better
  detection than with 1 frame, but frame-to-frame displacements are shortened by about 20 % (MSD at
  lag 1 underestimated). The diffusion coefficient from the slope of the MSD over 4 lags is preserved.
  Report D from the MSD slope, not from single jumps.
- **Faint dots in the background** (T = 3): below the detection threshold, no effect on the Jaccard
  index or on D. They come from the noise of the HIGH target. Use the same display range for all
  methods in figures.
- **PALMTracer pixel convention:** the center of pixel i is at coordinate i.
- **Acquisitions.** Fix the camera ROI and the split ROIs for a whole session; image beads at the
  start and at the end (2CFit, drift). Make the HIGH as clean as possible; set the LOW to the level of
  the data you want to denoise. CARE helps in a window: no gain if the LOW is already good, no
  recovery if the information is not in the LOW.