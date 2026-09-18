# Reproducibility notes

## Installation environment and manuscript provenance

The root `environment.yml` and pinned `requirements.txt` define the runtime
used for the new installation guide. They are not the original manuscript
environment. The historical export is preserved verbatim in
`docs/environment.original.yml`; it contains Linux build identifiers, a local
prefix, and multiple CUDA package generations. Do not use that export as a
portable installation command.

The original export records Python 3.10.14, JAX 0.4.10, Flax 0.6.8, and
Optax 0.1.7. The installation guide uses Python 3.12, JAX 0.4.30, Flax 0.8.5,
and Optax 0.2.3. The early-stopping wrapper handles both Flax return
conventions. This compatibility change preserves the model architecture and
the existing splitting, feature-pruning, and evaluation rules. Numerical
equivalence across dependency versions has not been established.

## Validation completed during repository preparation

On Windows x86_64 / Python 3.12.7 / JAX CPU:

- Installed pinned runtime packages in a fresh virtual environment; `pip check` passed.
- Checked imports, forward-pass shape `(2, 6)`, finite logits, and early-stopping updates.
- Prepared a synthetic 72-cell, 22,919-feature dataset into a holdout and five folds.
- Ran one fold for one epoch and completed final holdout evaluation.
- Compared a full model with masked inputs against pruned input/parameter arrays;
  predictions agreed within `rtol=1e-5, atol=1e-5`.
- Saved and restored those pruned parameters and the feature mask; predictions
  still agreed within the same tolerance.
- Validated the original 16 notebooks as notebook documents and parsed their Python cells.
  This is a source check, not execution against their external datasets.

See `docs/requirements.windows-cpu.lock.txt` for the exact installed packages.
Run `python check_environment.py` from the repository root for the provided
environment check. Reviewer training uses the complete `data/Heart.hdf5`.

A separate [fresh-clone reviewer run](reviewer-validation.md) subsequently
verified complete Heart download, independent Conda and pip installations,
five-fold preparation, fold 0 training through natural early stopping, actual
iterative pruning to one retained feature, and checkpoint restore. Not verified:
a complete five-fold training run,
Linux/GPU execution, or manuscript result equivalence.
A subsequent [figure audit](../figure/VALIDATION.md) validated 17 organized
notebooks and parsed four R scripts; it executed the complete plasma notebook
and CellPhoneDB helper with local cached inputs. The other figure analyses
have not been executed against their manuscript datasets.
The optional figure requirements are not a validated version lock.

## Experimental behavior

- The initial holdout split uses seed 20201212. It reserves the first shuffled
  sample of each class for training and the second for testing, then fills
  the requested test size. It is not a proportional stratified holdout; its
  actual size may differ from the requested ratio.
- `--random_state` controls only the stratified CV split, whose default is 42.
- The training seed initializes JAX. The shuffled PyTorch DataLoader is not
  explicitly seeded, so repeated runs are not guaranteed to agree.
- Validation and test metrics average batch means without weighting the final
  short batch by its sample count.
- Final testing uses the last training state and feature mask, without
  restoring the best validation checkpoint.
- Each fold evaluates the same holdout. These are not independent test
  cohorts, and test scores should not guide hyperparameter selection.
- Cell-level splitting does not enforce donor separation. The intended
  evaluation unit must be checked against the manuscript and donor metadata.

## Before reporting manuscript results

Record the dataset accession, feature identifiers and order, preprocessing,
label/covariate meanings, seeds, commands, hardware, and final package versions.
Validate the intended checkpoint-selection and metric-aggregation procedure.
Passing an installation or synthetic training check does not reproduce the
manuscript results.
