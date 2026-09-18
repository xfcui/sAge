# Execution validation record

## Quick run check on 2026-09-17

A fresh checkout of `qianminbio/sAge` at `2d69f63`, with the local Quick run
documentation and runner changes, downloaded the complete Heart file with
Git LFS. Its SHA-256 matched the value below. This check reused the existing
pinned Python 3.12.14 Conda environment on Windows / JAX CPU; it did not
perform a fresh dependency installation. Both `python -s -m pip check` and
`python -s check_environment.py` passed.

The README preparation command produced 3,104 cells, the 2,484/620 initial
split, and five CV folds. The fold 0 dry run passed. One epoch with the
default training settings then exited successfully and completed holdout
evaluation (`#Final Test: Loss = 1.6727, Acc = 31.33%`). These are execution
diagnostics, not manuscript results. The runner preserved `-s` in the child
command, used unbuffered logging, and printed the log path and completion
message. A mocked subprocess check also verified argument forwarding,
command recording, and error propagation via `check=True`.

This check did not repeat pruning, all-five-fold training, GPU execution,
or figure generation. The earlier records below retain their original scope.

## Fresh check after the public URL changed

On 2026-09-16, commit `7b0523a` was cloned from
`https://github.com/qianminbio/sAge.git` into a new checkout. Git LFS fetched
the complete `data/Heart.hdf5` from that repository: 284,573,664 bytes, with
the SHA-256 stated in `data/README.md`. An isolated Python 3.12 environment
previously installed from this repository's pinned model requirements passed
`pip check` and `check_environment.py` on JAX CPU. This check reused the
environment; it did not reinstall packages from scratch after the URL change.

The README data-preparation command produced 2,484 initial training cells,
620 held-out test cells and five CV folds. The documented dry run found the
prepared fold files. Fold 0 then completed one epoch and final holdout
evaluation, saving `command.json` and `train.log` (final diagnostic loss
1.6728; accuracy 31.33%). The quick-run variant of `--dry-run --folds 0`
also succeeded. These metrics demonstrate execution only; they are not
manuscript results. The model code and Heart example were unchanged by the
repository-owner rename, and the longer pruning run below remains the
evidence for iterative pruning. This check did not rerun five full folds,
GPU/Linux execution or the figures.

## Earlier full-fold-zero execution audit

This record tests whether the public `qianminbio/sAge` repository can execute
the model from its README on the complete Heart example. It is an execution
check, not a reproduction of every manuscript result.

The execution audit was conducted before the owner renamed their GitHub
account. The current repository has the same code history and Heart example;
the namespace change itself did not alter the model or dataset.

## Fresh clone and environment

On 2026-09-15, the public repository was cloned afresh on Windows x86_64 and
Git LFS downloaded `data/Heart.hdf5` (284,573,664 bytes). Its SHA-256 matched
the value in `data/README.md`:

```text
5a72f755adb1ed9eba62d85c7ab2d2f3502150be01d0892f4d7dd335b347e27f
```

The file opened as HDF5 with `data` shape `(3104, 22919)` and `label` shape
`(3104, 3)`. Two independent Python 3.12 environments were built from the
published `requirements.txt`: a fresh virtual environment and a fresh Conda
environment from the root `environment.yml`. `pip check` passed in each
environment when the Conda command used `python -s`. The installation checker
printed JAX CPU devices and completed a `(2, 6)` model forward pass.

The preparation machine had a broken user-configured conda-forge mirror.
Using the official conda-forge URL in `environment.yml` allowed Conda creation
to finish. Its Windows Python user-site directory also held unrelated broken
packages; `python -s` isolated the Conda checks and training from those.

## README workflow on complete Heart

From the fresh clone, the documented data-preparation command created an
initial train/test split of 2,484/620 cells and five cross-validation folds.
The documented `--dry-run` discovered all five fold file sets. The documented
single-fold, one-epoch check completed training and final holdout evaluation
with both a fresh Conda environment and the pinned pip environment.

The same full Heart input then completed fold 0 for eight epochs with the
default model heads, batch size, repeat count, learning rate, and masking
settings. The training log recorded feature counts of 22,919 at epoch 5,
21,774 at epoch 6, and 20,686 at epoch 7. It wrote `feature.txt` masks and
checkpoints in corresponding `featureXXXXXX` directories. The 20,686-feature
checkpoint was restored in the fresh pip environment and produced finite
`(2, 6)` logits for two held-out Heart cells.

This verifies the small-sample concern: the demonstration uses the complete
Heart file, and pruning was observed in an actual training run. A one-epoch
check alone is too short to exercise pruning.

After the eight-epoch check, an independent fold 0 run used the README's full
training setting `--max_epochs 9999` and the default early-stopping rule. It
actually reached nine retained features at epoch 163, then one at epoch 171.
Early stopping ended the run at epoch 234, and final holdout evaluation
completed. Checkpoints were saved for feature counts 9 through 2. No
one-feature checkpoint was saved because the validation score after that
pruning step fell below the code's initialized 80% save threshold. The final
test used the last one-feature training state, not a restored checkpoint.

The eight-epoch 20,686-feature count was only an intermediate observation.
With the complete Heart file, the code did continue pruning to single digits.

## Limits

The long run used only fold 0. It does not establish the scores from all five
folds, manuscript numerical equivalence, or GPU/Linux execution. Its accuracy
is an execution diagnostic and must not be presented as a paper result. The
PyTorch loader shuffle is not explicitly seeded, so scores and exact pruning
epochs can vary across runs. Figure analysis and the full benchmark suite have
separate inputs and were not executed in this audit.
