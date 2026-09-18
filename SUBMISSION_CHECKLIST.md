# Submission release checklist

## Scientific provenance

- [ ] Confirm these model, feature-selection, and evaluation settings match the manuscript.
- [x] Confirm model name: sAge.
- [ ] Add manuscript title, authors, and citation metadata.
- [x] Document the original age, method and tissue ID mappings.
- [ ] Document all datasets, accessions, preprocessing and feature order.
- [ ] Confirm holdout and CV splitting units and whether donor separation is required.
- [ ] Resolve or explicitly justify the evaluation and random-seed limitations in docs/reproducibility.md.
- [ ] Supply commands/configurations linking each reported result to its dataset and seed.

## Reproduction

- [x] Add a separate installation environment with pinned runtime dependencies.
- [x] Validate CPU installation, synthetic one-epoch training/evaluation, and a pruned-checkpoint round trip.
- [ ] Recover and validate the original manuscript environment and establish result equivalence.
- [x] Verify data preparation, one training epoch, pruning, checkpoint save/restore,
      and evaluation in a clean environment.
- [ ] Reproduce the manuscript experiments and preserve logs/configurations.
- [ ] Document hardware and expected runtime based on actual runs.
- [ ] Provide data and any released weights separately with checksums.
- [x] Include the complete Heart reviewer example via Git LFS.
- [x] Confirm Heart label mappings from the original repository notes.
- [ ] Confirm Heart data provenance, gene order and public redistribution rights.
- [ ] Inventory and release the actual training/evaluation code for every central benchmark result.
- [x] Supply the CellPhoneDB helper and figure input-path inventory.
- [ ] Add dataset accessions, checksums, and access routes for the missing figure inputs.
- [x] Validate 17 notebook documents/Python cells and parse four R analysis scripts.
- [x] Execute the plasma notebook and CellPhoneDB helper with local cached inputs.
- [ ] Execute the remaining figure analyses with released inputs and lock their final dependencies.

## GitHub handoff

- [x] Confirm that the submission does not require anonymity.
- [x] Specify the public GitHub repository: https://github.com/xfcui/sAge.
- [x] Retain the original repository's MIT license and copyright notice.
- [ ] Review third-party attribution and separate dataset permissions.
- [x] Provide a portable environment.yml; retain the original export in docs/environment.original.yml.
- [x] Review published files; generated outputs are ignored and Heart is explicitly tracked by Git LFS.
- [ ] Complete the README release metadata and remove resolved preparation notes.
- [ ] Tag the exact version used for submission and include that link in the manuscript.

Repository: https://github.com/xfcui/sAge. Manuscript release readiness
depends on the unchecked scientific and reproduction items above.

See [author-supplied release information](docs/release-information.md) for the remaining metadata and scientific decisions.
