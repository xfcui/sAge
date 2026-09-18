# sAge

**Quick navigation:** [Quick run](#quick-run) ·
[Installation](#2-install-the-environment) ·
[Heart example](data/README.md) · [Training](#4-train-the-model) ·
[Figure workflow](figure/WORKFLOW.md) ·
[Validation record](docs/reviewer-validation.md) ·
[Release information](docs/release-information.md)

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

If using the Conda option on a machine with user-installed Python packages,
replace `python` with `python -s` in these three commands as described in the
installation section. Leave several gigabytes free for the generated splits
and outputs; the Heart file itself is about 285 MB.

The data-preparation command should report **3,104 cells**, a **2,484/620**
initial train/test split, and **five CV folds**. The dry run prints the resolved
training command and checks file locations. The training runner writes its
output to `outputs/Heart-quick-check/fold_0/train.log`, including the final
holdout evaluation. The runner prints the log location at startup and a
completion message when training finishes. A successful run exits without an
error and its log contains `#Final Test: Loss = ..., Acc = ...`. Use a **new**
output directory for each repeat. For pruning and complete experimental
settings, continue with [the training section](#4-train-the-model). The
[execution record](docs/reviewer-validation.md) states exactly what has been
validated. Figure notebooks require separate inputs described in
[the figure workflow](figure/WORKFLOW.md).

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

PyTorch supplies the data loader; JAX performs model computation.
These are installation-guide versions, **not a claim about the software used
to produce the manuscript results**. The original environment export is
preserved in [docs/environment.original.yml](docs/environment.original.yml).

Verified from a fresh public GitHub clone on Windows x86_64 / JAX CPU: Git LFS
Heart download and checksum, independent Conda and virtual-environment installs,
five-fold data preparation, one-fold training with final test evaluation, and
an eight-epoch run that actually pruned features and saved a restorable checkpoint.
See [the execution validation record](docs/reviewer-validation.md) for exact scope and
results. An earlier Windows CPU package snapshot is available at
[docs/requirements.windows-cpu.lock.txt](docs/requirements.windows-cpu.lock.txt).
GPU execution has not been independently tested.

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
from the original [dataset notes](data/readme.txt). Gene order, data accession,
preprocessing provenance, and redistribution rights still need documentation.

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

To exercise the pruning branch without starting all five folds, use a new
output directory for a longer fold 0 run:

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-pruning-check --folds 0 -- --max_epochs 8
```

Pruning depends on validation improvements, so the exact epoch and retained
feature count can vary. The [execution validation record](docs/reviewer-validation.md)
shows both this short check and a complete fold 0 run that reached single-digit
features. Use the full run below to observe the training stop condition.

### Run all five folds

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart -- --seed 20201212 --max_epochs 9999
```

Folds run sequentially. Training output is written to each fold's
`train.log` with unbuffered output so you can follow progress while it runs.
The runner stops on an error and rejects nonempty fold output directories;
choose a new output directory when rerunning an experiment.

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

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate the selected environment; install with `python -m pip install -r requirements.txt`. |
| Conda mirror returns HTTP 404 | Use the official channel URL in `environment.yml` and retry; Option B avoids Conda channels. |
| `pip check` lists unrelated user packages | In Conda, use `python -s -m pip check` and `python -s` for later commands. |
| Dependency/version error | Create a fresh environment using the pinned requirements; avoid mixing with the historical export. |
| JAX lists only a CPU | Check the operating system, NVIDIA driver, and CUDA backend; run `check_environment.py --require-gpu`. |
| Feature-count assertion | The input must have 22,919 features in the expected order. |
| Missing fold file | Run data preparation first and pass its output directory as `--data-dir`. |
| Output directory is not empty | Choose a new `--output-dir`; existing experiments are retained. |
| Training fails after the runner prints a command | Inspect that fold's `train.log` for the underlying error. |

## 6. Figure analysis

The `figure/` directory contains the manuscript analysis notebooks and R scripts.
These require additional dependencies and external input/results files. Start
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
| `run_cross_validation.py` | Run multiple folds and save logs |
| `check_environment.py` | Check imports, devices, model execution, and early-stopping compatibility |
| `environment.yml`, `requirements.txt` | Installation environment |
| `docs/environment.original.yml` | Historical environment export |
| `docs/reproducibility.md` | Validation scope and experimental caveats |
| `docs/reviewer-validation.md` | Fresh-clone execution validation record |

## Reproducibility and citation

See [reproducibility notes](docs/reproducibility.md) for validation status,
data-splitting behavior, random seeds, checkpoint selection, and metric
aggregation. The manuscript title, author list, citation and data accession
are pending. The original [MIT license](LICENSE) is retained. See
[the submission checklist](SUBMISSION_CHECKLIST.md)
for the remaining release items.

The available MLP benchmark notebooks are in
[Figure 2](figure/figure2/), with required inputs listed in
[the figure input inventory](figure/INPUTS.md). The manuscript's full benchmark
model list and the scripts/settings behind every reported comparison still
need author review before claiming that all benchmark results are reproducible.
