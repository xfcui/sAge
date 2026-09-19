# Submission workflow and original sources

The submission workflow was imported from `qianminbio/sAge` at commit
`c521347194b9936ef7ef5b645e43473f8018baa1`, on a branch of `xfcui/sAge`
based on commit `9f4ca3acaa4b120f6f0fe8b29c54b74b33ab68c3`.
The original repository's history and MIT license are retained.

## Entry points

| Location | Purpose |
| --- | --- |
| `model/`, `run_cross_validation.py` | Documented Heart preparation, cross-validation and training workflow |
| `src/` | Original model, data loader and training program, preserved without edits |
| `data/convert.py` | Original conversion from external `hbchen.hdf5` to `xfcui.hdf5` |
| `data/readme.txt` | Original dataset shapes and age/method/tissue encodings |
| [Original README](original-README.md) | Preserved instructions for the original `src/` program |
| `figure/` | Organized figure analyses and their external-input inventory |

The submission model disables the tissue embedding that is active in the
original `src/model.py`. It uses prepared fold files and provides an
early-stopping compatibility wrapper for the documented dependency versions.
The original and submission programs must not be described as numerically
equivalent. Follow the main README for the submitted workflow, and consult
[reproducibility notes](reproducibility.md) for evaluation and seed behavior.

The original conversion data and full combined dataset are not bundled.
The complete Heart example is tracked by Git LFS. Figure inputs are separately
listed in [the inventory](../figure/INPUTS.md) and are not all released.

## Reviewing the update branch

Until the pull request is merged, clone the update branch explicitly:

```bash
git clone --branch submission-update --single-branch https://github.com/xfcui/sAge.git
cd sAge
git lfs pull
```

Then follow the installation and Quick run sections of the main README.
After merging, its normal clone command will obtain the updated default branch.
Historical execution checks in [the validation record](reviewer-validation.md)
retain the repository names and dates on which they were actually performed.
