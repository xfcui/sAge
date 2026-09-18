# Information needed for the manuscript release

This page distinguishes verified repository facts from information that must
come from the study authors. It is not a substitute for the paper's Methods.

## Heart data dictionary

The complete file is described in [the data guide](../data/README.md).

| Field | Verified representation | Biological definition / provenance |
| --- | --- | --- |
| `data` | 3,104 rows by 22,919 float32 features | Source accession and preprocessing pending |
| `label[:, 0]` | Class 1: 2,537 cells; class 4: 567 cells | Class names, ages and units pending |
| `label[:, 1]` | Constant ID 1; used by the model | Covariate name and ID mapping pending |
| `label[:, 2]` | Constant ID 6; unused by the model | Covariate name and ID mapping pending |
| Feature order | 22,919 positions; masks retain this order | Matching ordered gene identifiers pending |
| Cell / donor identity | Not stored in this HDF5 | Row-aligned identifiers and donor split policy pending |

An ordered gene file must have exactly one identifier per matrix column, in
the original order, without a header. Matching the line count alone does not
prove the order. Preserve the source feature table or preprocessing record
that establishes this correspondence. A `1` in `feature.txt` selects the gene
at that same position; do not sort the identifiers before applying the mask.

## Information to supply

- Manuscript title, ordered authors, affiliations and corresponding contact.
- Dataset accession, original download location, preprocessing commands and
  public redistribution terms for Heart and each released figure input.
- Confirmed class/covariate mappings and the ordered Heart gene identifiers.
- For each manuscript panel: source-data file, generating script, configuration,
  split, seed and expected output. Use [the figure workflow](../figure/WORKFLOW.md).
- Review third-party attributions and any separately licensed datasets.
  The original repository's [MIT license](../LICENSE), copyright 2026
  Xuefeng Cui, is retained.

## Experimental decisions requiring author confirmation

The [current implementation](reproducibility.md) uses cell-level splits,
unseeded PyTorch loader shuffling, means of batch-level evaluation metrics,
and the last training state for final testing. Confirm these choices against
the reported Methods, including whether donor separation is required.
Changing these choices can change the results and requires a new validated
experiment; documentation cleanup does not establish result equivalence.

## Freeze the submission version

After completing the metadata, source-data release and intended experiment
checks, create a versioned GitHub release and cite its immutable tag or commit
in the manuscript. Record the hardware, elapsed runtime, environment package
versions, commands and logs for that release. The repository's `main` branch
continues to change and is not a fixed manuscript version.
