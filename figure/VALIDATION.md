# Figure-code validation record

## Notebook launch and renamed files

After the Figure 5/6 file renames, documentation and notebook headings were
updated to the published names. All 17 notebook initialization cells were
executed in separate Python processes with each notebook's own directory as
the working directory. The repository lookup and `figure.paths` imports
succeeded in every case. Notebook-format and Python-syntax checks passed,
and local Markdown file links were checked. This verifies initialization;
it is not a rerun of the downstream scientific analyses.

## Analysis execution

On 2026-09-15, the optional `requirements-figures.txt` package set was
installed in a fresh Python 3.12 virtual environment that already held the
pinned model requirements. `python -m pip check` passed. All 17 organized
notebooks passed notebook-format validation, and their ordinary Python code
cells parsed successfully. All four R analysis scripts passed `parse()` with
R 4.3.3. This is a source and installation check, not figure reproduction.

The original `cellphonedb_interaction_count_network.py` helper was recovered
from the author's local analysis directory and included in `figure/figure5/`. With
`SAGE_FIGURE_INPUT_ROOT=E:/` and an ignored audit directory as the output
root, `python -m figure.figure5.cellphonedb_interaction_count_network` read the four
original small CellPhoneDB/source-evidence CSV files, completed its analysis,
and saved result tables and PNG/PDF/SVG network figures. After moving the
module into Figure 5, the same command was rerun with local input CSVs and
completed successfully (26 routes, 14 edges and 3 ligand–receptor pairs).
Only its paths, output-directory creation and package-relative import were
changed from the author's source version.

Every code cell in `figure/figure5/figure-5-1-plasma.ipynb` was then executed
in order using the same local external/cached inputs and isolated output
directory. It loaded 14 shared tissues, completed the age-stratified network,
age-specific ligand–receptor comparison, liver/plasma VTN panels, and receiver
cell panel, and saved the corresponding figure and source-data files. A path
refactor exposed one original ordering issue: the notebook wrote a route CSV
before creating its output directory. This was fixed before the successful
run.

The original input CSVs and generated audit files were **not** uploaded to
GitHub. Other figure notebooks and R analyses were not run against manuscript
datasets, and final panel mapping, numerical agreement, data accessions, and
redistribution permission still need author review. See [INPUTS.md](INPUTS.md)
for the external input paths that must be supplied for full reproduction.
