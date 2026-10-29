# Generating ICU patient ages with a GAN

JLab Data Science interview, Problem #2: a GAN in TensorFlow that reproduces the age distribution of 2,520
critical-care patients in `eICU_age.npy`. The data come from the
[eICU Collaborative Research Database](https://eicu-crd.mit.edu/about/eicu/), via
[JeffersonLab/jlab_datascience_data](https://github.com/JeffersonLab/jlab_datascience_data).

## Contents

| File | What it is |
|---|---|
| `eICU_age_gen_GANs.ipynb` | the full analysis, with all outputs and figures |
| `eICU_age.npy` | input data |
| `splits.npz` | fixed train / validation / test indices |
| `results/search_runs.csv`, `results/search_evals.csv` | the saved hyperparameter search, loaded instead of retraining |
| `models/final_generator.keras` | the final generator |

Running the notebook also writes all figures (`figures/`) and tables (`results/`).

## Approach

- **No data leakage:** a 60/20/20 split (1,512 / 504 / 504), stratified by age band. The training set fits
  everything, the validation set makes every choice, and the test set is scored once at the end.
- **Model:** a PacGAN, in which the discriminator judges packs of 4 ages, with the standard non-saturating loss.
  Training ages get uniform noise; generated ages are rounded to whole years.
- **Model selection:** Wasserstein distance + JS divergence, each divided by its noise floor, so a perfect model
  scores about 2. A simulation chose this score. A two-stage hyperparameter search and the best checkpoint are
  both chosen on validation.

## Result on the test set (504 patients)

- Score 2.22 ± 0.92. A perfect model scores 1.4–2.7 (95% range), so the GAN cannot be told apart from it.
- 4.4% of generated ages are 89, against 5.0% in the test set (the eICU "> 89" cap).
- The GAN clearly beats a normal fit and a KDE with Silverman's bandwidth. It is statistically tied with
  resampled training ages and with a KDE whose bandwidth was chosen on validation.

## How to run

Python 3.11, TensorFlow 2.18.1 (CPU), NumPy 1.26.4, pandas 2.3.3, SciPy 1.17.0, scikit-learn 1.9.0,
Matplotlib 3.10.1.

Open the notebook from this folder and run all cells (about 5 minutes). The saved search runs in `results/` are
loaded. To retrain the search (about 1.5 hours), delete `results/search_runs.csv` and `results/search_evals.csv`.

To generate ages from the saved generator without running the notebook (it outputs scaled ages, so convert them
with the training mean and standard deviation, then round and clip):

```python
import numpy as np, tensorflow as tf
G = tf.keras.models.load_model("models/final_generator.keras")
z = np.random.default_rng(0).standard_normal((100_000, 8)).astype("float32")
ages = np.clip(np.floor(G(z).numpy().ravel() * 17.700031 + 63.080688 + 0.5), 16, 90).astype(int)
```

## Use of AI tools

Claude Code (Anthropic, model Claude Opus 5.5) helped plan the analysis and write the code (see Section 0 of the
notebook).
