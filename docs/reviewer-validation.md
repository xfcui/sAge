# Execution validation record

## README-to-gene-selection check on 2026-10-28

The command-line entry point in `model/gene_selection.py` was also executed
on the same completed Heart fold. Its gene TXT/CSV, selected mask and feature
metrics CSV were byte-for-byte identical to the notebook exports. The README
now uses this command for direct results; Jupyter is optional. Command-line
plots use a noninteractive backend and do not require a display server.

The author-supplied `data/header.txt` contains 22,919 distinct nonempty gene
identifiers. It was copied without changes; its SHA-256 is recorded in
`data/README.md`. Its length matches Heart expression columns and saved masks;
this check alone does not establish the biological identity of each column.

The updated `model/elbow-gene-choose.ipynb` executed from top to bottom in the
Python 3.12.16 `sage` environment, using an existing completed local Heart
fold 0 run and the README directory layout. There were 104 saved candidate
masks with 4 through 1,011 genes. The default initial train/test selection
mode used 2,484/620 cells and exported 12 genes. A separate validation-mode
execution used 1,987/497 cells and exported 42 genes. These counts describe
this execution check, not validated manuscript gene lists or independent
test-set accuracy. The completed training run was reused, not retrained here.

The executed notebook retained its plot and table outputs. Gene-list indices,
header names and mask positions were reconciled. The PNG plot was visually
inspected, including the full candidate range and a separate elbow-search view.
Twelve unit tests passed, covering checkpoint compatibility, invalid masks,
header ordering, insufficient/flat curves, age mapping and a known segmented
breakpoint. `pip check` and notebook-format validation passed. Installed
selection packages included NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.5.1,
h5py 3.11.0, Matplotlib 3.11.2, pandas 2.3.3 and ipykernel 6.31.0.

Correction: an earlier version of this record incorrectly stated that the
original notebook squared the sum of residuals. Inspection of the original
notebook confirms that it already sums squared residuals. The current version
uses the same objective. The default selection mode retains the previous
initial-test usage and explicitly labels it as selection data; the optional
validation mode leaves that holdout unused by the notebook. No manuscript
numerical-equivalence claim is made.

The author-supplied original notebook was compared against the updated code
on the same local Heart fold 0 masks, initial train/test data and header. The
original notebook's mapping, mask parsing, regression and elbow functions
were extracted for this check; it was not a full original-notebook execution.
Both calculations used one BLAS thread. Initially only 51 of 104 MAEs agreed
within `rtol=1e-5, atol=1e-5`; one candidate differed by about 1,922 months.
For a 260-feature candidate, the original pandas selection produced a C-order
matrix while NumPy boolean selection produced a Fortran-order matrix. Although
the values were identical, the ill-conditioned fit reported ranks 253 and 254
and substantially different metrics. The current code now converts selected
matrices to C-order, matching the original layout.

After that change, all 104 MAEs, RMSEs and PCCs matched exactly in this
environment, and both implementations selected the same 12-feature mask and
gene identifiers (ignoring the original notebook's uppercasing). A unit test
checks C- and Fortran-order inputs. This comparison does not establish
equivalence for other folds, machines, environments or manuscript results;
ill-conditioned regression remains sensitive to numerical conditions.

## Windows checkpoint fix on 2026-10-27

The local `xfcui/sAge` checkout at `9b378ad` failed in fold 0 after epoch 10.
The traceback ended in Orbax 0.6.4's `_log_io_per_sec_metric`, where a zero
wall-clock interval produced NaN and `humanize.naturalsize` raised
`ValueError: cannot convert float NaN to integer`. This was a checkpoint
diagnostic formatting failure, not evidence of NaN training loss.

The compatibility fix replaces only that version's throughput logger and
omits unmeasurable rates. It leaves serialization and the checkpoint format
unchanged. A controlled zero-duration test reproduced the original exception
and passed after the fix. Three regression tests cover invalid durations,
valid throughput, and installation of the compatibility handler.

In the user's existing Windows CPU Conda environment (Python 3.12.16,
Orbax 0.6.4), the expanded environment check saved and restored model
parameters and a feature mask. With the fix, complete Heart fold 0 finished
12 epochs, saved checkpoints through 16,851 retained features, and completed
holdout evaluation (diagnostic loss 0.1618; accuracy 96.15%).

A separate unmodified checkout at the same commit completed all five folds
for one epoch each using the earlier pinned environment and prepared Heart
splits. This verifies the five-fold runner path, not full five-fold pruning.
Neither check establishes manuscript scores or full five-fold convergence.
Failed output directories were retained; the fixed run used a new directory.

## xfcui submission branch check on 2026-09-18

The public `xfcui/sAge` branch `submission-update` at commit `3d9f35b` was
cloned into a new checkout. Git LFS downloaded `data/Heart.hdf5` from the
teacher-owned repository (284,573,664 bytes); its SHA-256 was
`5a72f755adb1ed9eba62d85c7ab2d2f3502150be01d0892f4d7dd335b347e27f`.

A new Python 3.12.7 virtual environment was created and installed from the
branch's `requirements.txt`. `pip check` passed. `check_environment.py` passed
on JAX CPU, including the `(2, 6)` model forward pass and early-stopping API.
The README preparation command produced 3,104 cells, a 2,484/620 initial
train/test split and five CV folds. The documented fold 0 dry run passed.
The one-epoch Quick run then completed training and holdout evaluation in
30.6 seconds on the validation machine (`#Final Test: Loss = 1.6727,
Acc = 31.33%`). It wrote the documented `command.json` and `train.log`.

This establishes that the code, dependencies and Git LFS Heart object can be
obtained from the teacher-owned update branch and that the Quick run completes.
It is an execution check, not a reproduction of manuscript scores. The full
five folds, long pruning run, GPU/Linux path and figure analyses were not
repeated in this branch check; their earlier validation scopes remain below.

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
