## sAge

Feature selection for single-cell age classification with a masked pruning model.

The model predicts donor age from gene-expression profiles while iteratively dropping genes that contribute least to accuracy. Training is implemented in JAX / Flax.

## Requirements

- Python 3
- JAX, Flax, Optax
- NumPy, h5py, tqdm
- PyTorch (used only for `DataLoader`)
- A CUDA GPU is expected for full training

Activate the existing `jax` conda environment, or install the same stack in a new environment.

## Data

Training reads `data/xfcui.hdf5`:

| Dataset | Shape | Contents |
| --- | --- | --- |
| `data` | `(290905, 22919)` | gene counts |
| `label` | `(290905, 3)` | age, method, tissue |

Label encodings:

- **age** (`label[:, 0]`): `1m=0`, `3m=1`, `18m=2`, `21m=3`, `24m=4`, `30m=5`
- **method** (`label[:, 1]`): `droplet=0`, `facs=1`
- **tissue** (`label[:, 2]`): 19 tissues (Fat, Bladder, Brain, Heart, Kidney, …)

If you only have the compressed file:

```bash
gunzip -k data/xfcui.hdf5.gz
```

To rebuild the HDF5 from `hbchen.hdf5`, run `data/convert.py` from `data/`. Tissue-level source shapes are listed in `data/readme.txt`.

## Training

From the repository root:

```bash
python3 src/train.py --save debug
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--numhead` | `8` | number of attention / feature heads |
| `--tgtacc` | `80` | validation accuracy that must be held while pruning |
| `--maskrate` | `0.15` | fraction of genes masked in each training view |
| `--batchsize` | `64` | training batch size |
| `--batchrepeat` | `8` | repeated masked views per batch |
| `--learnrate` | `1e-3` | peak learning rate after warmup |
| `--weightdecay` | `1e-3` | LAMB weight decay |
| `--decayrate` | `0.95` | per-epoch LR decay |
| `--patience` | `64` | early-stopping patience |
| `--save` | none | directory for Flax checkpoints |
| `--seed` | `20201212` | RNG seed |

Each improved epoch with `--save` writes a checkpoint under `featureNNNNNN/`, including the pruned parameter tree and the boolean gene mask.

## How pruning works

1. Inputs are library-size normalized and log-transformed.
2. Training applies random gene masks and conditions on method and tissue embeddings.
3. After warmup, genes with the weakest masked-accuracy signal are dropped (about 5% per round).
4. Training continues until validation accuracy stays below `--tgtacc` for `--patience` epochs.

## Layout

```
src/           training code
  train.py     MaskedPruningModel training and pruning loop
  model.py     Flax model
  data.py      HDF5 dataset and train/valid split
data/          dataset notes and conversion script
```
