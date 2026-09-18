# Heart example data

See [the data dictionary and release information](../docs/release-information.md)
for verified label counts and the biological definitions still to be supplied.

The original repository's [dataset notes](readme.txt) document the source
encoding: age IDs 0–5 denote 1, 3, 18, 21, 24 and 30 months; method IDs
0/1 denote droplet/FACS; tissue ID 6 denotes Heart. Under that encoding,
the example's age IDs 1/4 mean 3/24 months and method ID 1 means FACS.
These notes supply the historical interpretation; the exact provenance and
feature-order correspondence of the released file still need documentation.
The original [conversion script](convert.py) requires `hbchen.hdf5`, which
is not bundled and is not needed for the Quick run.

The example is the **complete** `Heart.hdf5` dataset. It contains
3,104 cells with 22,919 float32 expression features. It is stored with
Git LFS because the file is approximately 285 MB.

| HDF5 key | Shape | Meaning |
| --- | --- | --- |
| `data` | `(3104, 22919)` | Cell-by-feature expression matrix |
| `label` | `(3104, 3)` | Class ID and two covariate columns |

Class IDs are 1 (2,537 cells) and 4 (567 cells); the model has six output
logits, of which only classes 1 and 4 occur in this file. The first covariate
has ID 1 for every cell. The second has ID 6 and is currently unused.
Expression values are nonnegative and finite.

Gene names, cell IDs, donor IDs, tissue labels, and a public data accession
are not stored in this HDF5. The author must document the source, feature
order, class/covariate meanings, and redistribution rights before final
manuscript release.

## Download and verify

Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/xfcui/sAge.git
cd sAge
git lfs pull
```

The expected SHA-256 of `data/Heart.hdf5` is:

```text
5a72f755adb1ed9eba62d85c7ab2d2f3502150be01d0892f4d7dd335b347e27f
```

If the local file is only a few bytes, Git LFS was not installed or fetched;
run `git lfs pull` from the repository. Allow several gigabytes of free disk
space for the generated holdout, five-fold splits, logs, and checkpoints.

## Run sAge on Heart

Install the environment as described in the
[main README](../README.md), then prepare the full dataset:

```bash
python model/prepare_dataset_for_cv.py --data_path data/Heart.hdf5 --output_dir prepared_data/Heart --initial_test_size_ratio 0.2 --n_cv_splits 5 --random_state 42
```

Check the generated paths, then run the five folds:

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart --dry-run
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart -- --seed 20201212 --max_epochs 9999
```

The runner writes each fold's `train.log` and feature masks/checkpoints
under `outputs/Heart/fold_N/`. Use a fresh `--output-dir` for any repeat
run. Full CPU training may take a long time. An optional Linux GPU setup is
described in the main README, but GPU execution has not been independently
validated for this repository release.

To check only whether one epoch can execute on the **same complete Heart
file**, run fold 0 into a separate directory:

```bash
python run_cross_validation.py --data-dir prepared_data/Heart --output-dir outputs/Heart-one-epoch --folds 0 -- --max_epochs 1
```

This execution check will not demonstrate feature pruning. In the current
training code, pruning is considered after epoch 4 and requires a validation
accuracy improvement above the initialized 80% threshold. A full run is
necessary to observe whether and when features are removed. Results from
one epoch are not the manuscript results.

For a longer fold 0 execution check on the same full file, follow the
eight-epoch pruning-check command in the [main README](../README.md).
