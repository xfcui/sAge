# sAge

**Quick navigation:** [Quick run](#quick-run) ·
[Installation](#2-install-the-environment) ·
[Heart example](data/README.md) · [Training](#4-train-the-model) ·
[Figure workflow](figure/WORKFLOW.md) ·
[Validation record](docs/reviewer-validation.md)

**sAge** is a JAX/Flax model for classification and iterative feature selection
from single-cell expression data. This repository provides data splitting,
training, cross-validation, and feature-mask export.

Use `model/` and `run_cross_validation.py` for the workflow below. The
original `src/` implementation, data conversion script and dataset notes
remain available; see [repository integration notes](docs/repository-integration.md)
for their differences and the original documentation.

**Workflow:** install the environment → prepare an HDF5 dataset → create the
train/test and cross-validation splits → train → inspect logs and selected features.

## Quick run

This is the shortest route to check that the published **model** runs on the
complete Heart example. It checks execution and final holdout evaluation;
one epoch is too short to demonstrate feature pruning or reproduce paper scores.

1. [Clone the repository and download Heart with Git LFS](#1-download-the-code).
2. [Install the pinned Python 3.12 environment](#2-install-the-environment)
   and run `check_environment.py` as shown there.
3. From the `sAge` repository directory, run:

```bash
python model/prepare_dataset_for_cv.py --data_path data/Heart.hdf5 --output_dir prepared_data/Heart --initial_test_size_ratio 0.2 --n_cv_splits 5 --random_state 42
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-quick-check --folds 0 --dry-run
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-quick-check --folds 0 -- --max_epochs 1
```

### Scope, expected output and runtime

The Quick run is an execution test, not a complete experiment. It verifies
data preparation, model construction, one training epoch and holdout
evaluation. It does **not** complete iterative pruning, determine the final
aging-gene list, run all five folds to early stopping or reproduce manuscript
scores.

The preparation command should report **3,104 cells**, a **2,484/620** initial
train/test split and **five CV folds**. A successful training check writes
`outputs/Heart-quick-check/fold_0/command.json` and `train.log`; the log ends
with `#Final Test: Loss = ..., Acc = ...%`. These values are execution
diagnostics.

After installation and the Heart download, the three commands typically take
approximately **1–2 minutes** on a standard CPU desktop. One validated Windows
x86_64 / JAX CPU run completed the one-epoch training and holdout evaluation
in **30.6 seconds**. Runtime varies with CPU, storage and available memory.

When using the Conda option, replace `python` with `python -s` in these commands.
Use a new output directory for every repeat. Leave several gigabytes free for
the generated splits and outputs; the Heart file itself is about 285 MB.

## 1. Download the code

Install Git and Git LFS, then run:

```bash
git lfs install
git clone https://github.com/xfcui/sAge.git
cd sAge
git lfs pull
```

Run the commands below from this repository directory. In Windows, use
**Anaconda Prompt** for the Conda commands.
Git LFS downloads the Heart example (approximately 285 MB).
See [the example-data guide](data/README.md).

## 2. Install the environment

### Option A: Conda (CPU)

```bash
conda env create -f environment.yml
conda activate sage
python -s -m pip check
python -s check_environment.py
```

The environment file creates a separate Python 3.12 environment and installs
the pinned dependencies in `requirements.txt` using the official conda-forge
channel URL. The `-s` flag keeps packages installed in the system's Python
user directory out of the Conda environment. Use `python -s` in place of
`python` in the data-preparation and training commands below when using Conda.
CPU execution is sufficient for the installation check; full experiments may
take substantially longer on CPU.

### Option B: Python virtual environment (CPU; no Conda required)

Install Python 3.12, then create a fresh environment from the repository root:

```bash
python -m venv .venv
```

Activate it on Windows (Command Prompt or Anaconda Prompt):

```bat
.venv\Scripts\activate.bat
```

Or activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Then run:

```bash
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python check_environment.py
```

`python --version` must show Python 3.12. Use **either Option A or Option B**.
The core versions are:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| JAX / jaxlib | 0.4.30 / 0.4.30 |
| Flax / Optax | 0.8.5 / 0.2.3 |
| NumPy / SciPy | 1.26.4 / 1.12.0 |
| PyTorch | 2.6.0 |
| h5py / scikit-learn | 3.11.0 / 1.5.1 |

### Typical installation time

On a standard desktop computer with a stable internet connection, creating a
fresh Conda or Python virtual environment and running `check_environment.py`
typically takes approximately **5–10 minutes**. The exact time depends mainly
on network speed and whether Conda or pip packages are already cached. This
estimate does not include cloning the repository or downloading the Heart
example with Git LFS.

### Tested system and hardware scope

The core model and Quick run were tested on **Windows 11 64-bit
(10.0.22631)** with Python 3.12 and the JAX CPU backend. No GPU or other
non-standard hardware is required for installation or the Quick run. Several
gigabytes of free disk space are recommended. The pinned root
`environment.yml` and `requirements.txt` cover the core sAge model and Quick
run. Dependencies under `figure/` are optional and are not required for this
software demo; their separate setup is documented in
[figure/README.md](figure/README.md).

PyTorch supplies the data loader; JAX performs model computation. A fresh
public-clone validation covered the Git LFS Heart download, independent Conda
and virtual-environment installation, five-fold data preparation, one-fold
training/evaluation, pruning and checkpoint restoration. See the
[execution validation record](docs/reviewer-validation.md) for the exact scope.

### Optional: NVIDIA GPU on Linux

First install the CPU environment above. On Linux with a working NVIDIA
driver, add the CUDA 12 backend for the same JAX version:

```bash
nvidia-smi
python -m pip install "jax[cuda12]==0.4.30"
python -m pip check
python check_environment.py --require-gpu
```

This GPU path has not been tested on the preparation machine. Native Windows
does not support the JAX NVIDIA GPU backend; use Linux or a suitable WSL2
setup. See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for platform and driver requirements. Do not install an unpinned latest JAX
over this environment.

A successful check prints package versions, the JAX devices, and a model
forward-pass result of shape `(2, 6)`. With `--require-gpu`, it fails if
JAX cannot see a GPU, so CPU fallback is not mistaken for GPU execution.

## 3. Prepare your input data

The example is the complete `data/Heart.hdf5` (3,104 cells),
provided via Git LFS. For its schema, checksum, and download instructions,
read [data/README.md](data/README.md).
Substitute another HDF5 path when using your own preprocessed dataset.

Each HDF5 file must contain:

| Key | Shape | Contents |
| --- | --- | --- |
| `data` | `(N, 22919)` | Numeric expression matrix: cells by features |
| `label` | `(N, 1 + C)`, `C >= 1` | Class label followed by covariates |

For this implementation:

- Class labels in `label[:, 0]` are integer IDs from **0 to 5**.
- The first covariate, `label[:, 1]`, is an integer ID **0 or 1**.
- The model uses that first covariate only; the tissue embedding is disabled.
- Feature order must stay identical across all splits and downstream analysis.
- The training script normalizes each row by its total (clipped to at least 1),
  then applies `log2(x * 1023 + 1)`. Use the input preprocessing intended for
  the experiment; do not apply this transform twice.

The current Heart file contains only class IDs 1 and 4 (3 and 24 months),
although the model has six output classes. Its first covariate is method ID 1
(FACS); its unused second covariate is tissue ID 6 (Heart). These mappings come
from the original [dataset notes](data/readme.txt). The released HDF5 does not
embed gene identifiers or provenance metadata; the Quick run uses matrix
positions, and compatible user inputs must preserve the same feature order.

Create a holdout test set and five cross-validation folds:

```bash
python model/prepare_dataset_for_cv.py --data_path data/Heart.hdf5 --output_dir prepared_data/Heart --initial_test_size_ratio 0.2 --n_cv_splits 5 --random_state 42
```

Expected layout:

```text
prepared_data/Heart/
├── initial_split/
│   ├── train.h5
│   └── test.h5
└── cv_folds/
    ├── fold_0/
    │   ├── train.h5
    │   └── valid.h5
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    └── fold_4/
```

Every fold directory contains both `train.h5` and `valid.h5`. The CV folds
are formed from the initial training set. Use a new preparation directory
for each split configuration because this command can overwrite split files.

## 4. Train the model

### Check the file paths

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart --dry-run
```

This checks that the files exist and prints commands. It does not load the
HDF5 contents or train the model.

### Run a short check on one fold

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-check --folds 0 -- --max_epochs 1
```

One epoch on the complete Heart file checks execution. Feature pruning can
only start after validation improvement and later training epochs; a one-epoch
run does not demonstrate pruning. See the full run below for the original
training settings, and do not interpret a short run's scores as paper results.

### Optional short pruning check — not a complete training run

To exercise the pruning and checkpoint branches without starting a complete
experiment, use a new output directory for an eight-epoch fold 0 run:

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-pruning-check --folds 0 -- --max_epochs 8
```

Pruning depends on validation improvements, so the exact epoch and retained
feature count can vary. This command does not train fold 0 to early stopping
and does not produce the final aging-gene list. Use the full run below for a
complete experiment.

### Run all five folds

To check all five folds before a full experiment, run one epoch per fold
in a new output directory:

```bash
python -s run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-five-fold-check --folds 0 1 2 3 4 -- --seed 20201212 --max_epochs 1
```

This checks fold orchestration and evaluation, not feature pruning or
checkpoint creation. `python -s check_environment.py` also tests checkpoint
saving and restoration. For the full experiment, use:

```bash
python -s run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart --folds 0 1 2 3 4 -- --seed 20201212 --max_epochs 9999
```

Folds run sequentially. Training output is written to each fold's
`train.log` with unbuffered output so you can follow progress while it runs.
The runner stops on an error and rejects nonempty fold output directories;
choose a new output directory when rerunning an experiment.

The **1–2 minute** estimate applies only to the Quick run. Complete training
continues until early stopping or the maximum epoch count and takes
substantially longer. A CPU fold can require tens of minutes or more, and the
five sequential folds can require several hours. An NVIDIA GPU is recommended
for full training. No complete five-fold runtime is claimed until all five
folds have been timed on the same system.

Arguments **before** `--` configure the runner. Arguments **after** `--`
are passed to `model/train-tissue.py`.

| Training option | Default | Meaning |
| --- | --- | --- |
| `--numhead` | 8 | Number of model heads |
| `--batchsize` | 64 | Number of original cells per training batch |
| `--batchrepeat` | 8 | Number of augmented copies per batch |
| `--maskrate` | 0.15 | Feature masking probability |
| `--learnrate` | 0.001 | Learning-rate schedule peak |
| `--patience` | 64 | Early-stopping patience parameter |
| `--max_epochs` | 9999 | Maximum epoch count |
| `--seed` | 20201212 | JAX random seed |
| `--num_workers` | 0 | DataLoader worker count |

For example, reduce memory usage with
`-- --batchsize 16 --batchrepeat 2`. Changing these settings can change the
experimental results.

## 5. Find the outputs

```text
outputs/Heart/fold_0/
├── command.json
├── train.log
└── checkpoints/
    └── featureXXXXXX/
        ├── feature.txt
        └── ... checkpoint files ...
```

- **command.json:** exact command used for this fold.
- **train.log:** training/validation metrics and the final test loss and accuracy.
- **feature.txt:** a mask in the original feature order; `1` means retained
  and `0` means removed. It is not a list of gene names.
- **Checkpoint files:** saved parameters and feature mask at qualifying
  validation improvements. Checkpoints are not guaranteed after a short run:
  the existing training code initializes the accuracy threshold at 80%.

The remaining folds have the same structure. Prepared splits and generated
outputs are excluded from Git; the complete Heart example file is included.

## 6. Figure analysis

The `figure/` directory contains optional manuscript analysis notebooks and R
scripts. They are outside the core software demo and require additional
dependencies and external input/results files. Start
with [figure/README.md](figure/README.md) for the organized script index,
installation, input inventory, and validation scope. The model environment
alone does not install all
figure dependencies.

## Repository layout

| File | Purpose |
| --- | --- |
| `model/model.py` | sAge architecture (`MaskedPruningModel`) |
| `model/data.py` | Dataset loading and splitting |
| `model/prepare_dataset_for_cv.py` | Prepare holdout and CV files |
| `model/train-tissue.py` | Train and evaluate one fold |
| `model/elbow-gene-choose.ipynb` | Select the final feature count and aging-gene list using one chosen cross-validation fold |
| `run_cross_validation.py` | Run multiple folds and save logs |
| `check_environment.py` | Check imports, devices, model execution, and early-stopping compatibility |
| `environment.yml`, `requirements.txt` | Installation environment |
| `docs/environment.original.yml` | Historical environment export |
| `docs/reproducibility.md` | Validation scope and experimental caveats |
| `docs/reviewer-validation.md` | Fresh-clone execution validation record |

## Reproducibility and citation

See [reproducibility notes](docs/reproducibility.md) for validation status,
data-splitting behavior, random seeds, checkpoint selection and metric
aggregation. The software is distributed under the [MIT license](LICENSE).

The available optional MLP benchmark notebooks are in
[Figure 2](figure/figure2/), with required inputs listed in
[the figure input inventory](figure/INPUTS.md). These figure analyses are not
part of the Quick run or its expected runtime.
