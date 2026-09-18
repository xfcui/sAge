# Running the manuscript analyses

Start with [installation and configuration](README.md) and the
[input-path inventory](INPUTS.md). The complete Heart example exercises sAge
training; it does not contain the other tissues, benchmark results or cached
tables needed for these figures. Obtain those inputs before running an analysis.

## Execution steps

1. Install the model and optional figure dependencies in the same environment.
2. Put the required files at the relative locations listed in `INPUTS.md` under
   `figure_inputs/`, or configure `SAGE_FIGURE_INPUT_ROOT` before starting Jupyter.
3. Start `jupyter lab` from the repository root and select the **Python (sAge)**
   kernel. Open the desired notebook, restart its kernel and run cells in order.
   Its first cell locates the repository even if the kernel starts in a nested
   figure directory. This locates code; it does not download missing inputs.
4. For R, install the packages in `README.md` and run the selected script with
   `Rscript` from the repository root. For example:
   `Rscript figure/figure4/1-mouse-gene-upset-tissue.R`.
5. Inspect generated files under `figure_outputs/` (or the configured output
   root). Some upstream steps also write derived inputs; use working copies.
   Retain logs, package versions, source-data checksums and the Git commit.

## Analysis order and outputs

These are dependencies visible in the source, not a claim of a fully automated
pipeline. Some intermediate files must be supplied separately. Final panel
letters must be matched to the submitted manuscript by the authors.

| Group | Start with | Inputs required before downstream plotting | Outputs / next step |
| --- | --- | --- | --- |
| Preprocessing | [Human preparation](data-%20preprocess.ipynb) | Original human h5ad files and metadata | Derived HDF5 and gene lists; verify feature order |
| Feature selection | [Elbow analysis](elbow-gene-choose.ipynb) | Training feature masks, result folders and gene header | Feature-count analysis / selected gene sets |
| [Figure 2](figure2/) | Mouse/human benchmark notebooks | Dataset splits, selected genes and comparator outputs | MLP result tables used by heatmap and boxplot notebooks |
| [Figure 3](figure3/) | Nonlinear trajectory notebooks | Tissue expression, ordered genes and selected-gene lists | Trajectory results; cluster/GO analysis also needs assignment tables |
| [Figure 4](figure4/) | Gene-ratio and overlap analyses | Mouse/human gene sets and enrichment summaries | Ratios, overlap plots and GO comparisons |
| [Figure 5](figure5/figure-5-1-plasma.ipynb) | Plasma notebook | Cached CellPhoneDB, plasma VTN and receptor tables | Tissue networks, ligand–receptor and VTN panels |
| [Figure 6](figure6/) | Clock notebook, then intervention notebooks | Training data, saved clocks, intervention matrices, cluster assignments and ortholog mapping | Intervention scores/plots; target-gene analysis additionally needs external Excel tables |

The [CellPhoneDB helper](figure5/cellphonedb_interaction_count_network.py) can also run
as `python -m figure.figure5.cellphonedb_interaction_count_network` from the repository
root once its four input CSVs are present. It reads cached CellPhoneDB results;
it does not run CellPhoneDB itself.

## What has been tested

See [the validation record](VALIDATION.md). The plasma notebook and helper
executed with the author's local cached inputs. Other notebooks have received
format/Python-syntax checks; R scripts have received parsing checks. Full
public-data reproduction remains incomplete until the external inputs and
their provenance are released.
