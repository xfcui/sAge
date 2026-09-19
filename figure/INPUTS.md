# Figure input inventory

The manuscript figure code uses data and intermediate results that are not bundled in this GitHub repository. This inventory lists the portable input locations used in source code; it does **not** assert that those files are available or that each figure has been reproduced. The author must add data accessions, file checksums, and the final manuscript panel mapping.

## Configure paths

From the repository root, copy required inputs under `figure_inputs/` using the relative layouts below. `figure_inputs/` and `figure_outputs/` are ignored by Git. Alternatively, set `SAGE_FIGURE_INPUT_ROOT` to the directory containing these layouts and `SAGE_FIGURE_OUTPUT_ROOT` to a writable output directory. For example, on the original Windows preparation machine, the input root is `E:/`; on WSL it is `/mnt/e`. These examples describe where the files existed during analysis; a reviewer must obtain the data separately.

Run Jupyter from the repository root so `from figure.paths import ...` works. Run R scripts from the repository root so `source("figure/paths.R")` works. Relative paths and output names retain the source workflow labels. Some upstream notebooks also write derived files in their input layout; work from a copy of original data instead of pointing their input root at a source archive you need to preserve.

## Paths referenced by each analysis

Some listed paths are generated intermediates rather than primary datasets. Follow the code cell order and supply any upstream results before rerunning a downstream figure.

### `figure/figure5/cellphonedb_interaction_count_network.py`

- `0-figure-code/0-result-6-2-plasma-data-output`

### `figure/data- preprocess.ipynb`

- `2-8.3-shanda/1-data/GSE201333_RAW/`
- `2-8.3-shanda/1-data/GSE201333_RAW/2-human-tissue-filter-remove-AL-all-tissue`
- `2-8.3-shanda/1-data/GSE201333_RAW/2-human-tissue-filter-remove-AL-all-tissue.hdf5`
- `2-8.3-shanda/1-data/GSE201333_RAW/4-human-tissue-filter-remove-AC-all-tissue.txt`
- `2-8.3-shanda/1-data/GSE201333_RAW/GSM6058681_TabulaSapiens_filtered.h5ad`
- `2-8.3-shanda/1-data/GSE201333_RAW/GSM6058681_TabulaSapiens_no_clones.h5ad`
- `2-8.3-shanda/1-data/GSE201333_RAW/gene_names_no_clones.txt`

### `figure/elbow-gene-choose.ipynb`

- `1-TMS-remove/2-restart`
- `2-8.3-shanda/1-feature/1-5x`
- `header.txt`

### `figure/figure2/figure-2-1-mouse-benchmark-MLP.ipynb`

- `1-TMS-remove/2-restart`
- `1-TMS-remove/4-precision-5_to_25/1-5_to_25-pcc-add-model-mlp-tanh-parallel`
- `1-TMS-remove/kfold_results`
- `1-deeplearn/2-iAge/1-output/2-gene-Mapped_Results`
- `2-8.3-shanda/1-feature/1-5x`
- `2-8.3-shanda/1-gene-benchmark`
- `2-8.3-shanda/header.txt`
- `3-scimmuaging/output_5x`
- `6-buckley/1-5x`
- `8-XGBoost_Paper/output`
- `header.txt`

### `figure/figure2/figure-2-2-mouse-heatmap-MLP.ipynb`

- `1-TMS-remove/4-precision-5_to_25/1-5_to_25-pcc-add-model-mlp-tanh-parallel/1-end-mlp_tanh_random_sample_summary_parallel.csv`
- `1-TMS-remove/4-precision-5_to_25/1-5_to_25-pcc-add-model-mlp-tanh/1-end-mlp_tanh_random_sample_summary.csv`

### `figure/figure2/figure-2-3-mouse-boxplot-MLP.ipynb`

- `1-TMS-remove/4-precision-5_to_25/1-5_to_25-pcc-add-model-mlp-tanh-parallel`

### `figure/figure2/figure-2-4-human-benchmark-MLP.ipynb`

- `1-human-benchmark-model-output/0-gene-database`
- `1-human-benchmark-model-output/1-XGBoost_Paper/1-output`
- `1-human-benchmark-model-output/2-iAge/2-map-output`
- `1-human-benchmark-model-output/3-human-tissue-filter-remove-AL-all-tissue`
- `1-human-benchmark-model-output/3-scale/kfold_results-split_by_tissue`
- `1-human-benchmark-model-output/4-scimmuaging/output_5x`
- `1-human-benchmark-model-output/5-buckley/1-output`
- `1-human-benchmark-model-output/summary_age_prediction/mlp_tanh`
- `2-8.3-shanda/1-data/GSE201333_RAW/gene_names_no_clones.txt`

### `figure/figure2/figure-2-5-human-heatmap-MLP.ipynb`

- `1-human-benchmark-model-output/summary_age_prediction/mlp_tanh/human_benchmark_mlp_tanh_prediction_summary.csv`

### `figure/figure3/figure-3-1-nonlinear-trajectory.ipynb`

- `2-8.3-shanda/1-feature/9-Final_Segmented_Genes`
- `aging/scage/data/1-tissue-data`
- `header.txt`

### `figure/figure3/figure-3-2-cluster-GO.ipynb`

- `2-8.3-shanda/1-feature/10-2-0-Gene_Trajectory_Clusters_Annotated/`

### `figure/figure3/figure-3-3-human-nonlinear-trajectory.ipynb`

- `2-8.3-shanda/1-data/GSE201333_RAW/2-human-tissue-filter-remove-AL-all-tissue`
- `2-8.3-shanda/1-data/GSE201333_RAW/gene_names_no_clones.txt`
- `2-8.3-shanda/1-feature/1-human-guaidian-choose-gene/Gene_Lists`

### `figure/figure4/1-mouse-gene-GO-jaccard-similarity.R`

- `2-8.3-shanda/1-feature/9-Final_Segmented_Genes`

### `figure/figure4/1-mouse-gene-upset-tissue.R`

- `2-8.3-shanda/1-feature/1-human-guaidian-choose-gene/Gene_Lists/`
- `2-8.3-shanda/1-feature/9-Final_Segmented_Genes/`

### `figure/figure4/1-mouse-multitissue-GO.R`

- `2-8.3-shanda/1-feature/11-1-Enrichment_R_Results`

### `figure/figure4/figure-4-1-gene-ratio-different-tissue.ipynb`

- `2-8.3-shanda/1-feature/1-human-guaidian-choose-gene/Gene_Lists`
- `2-8.3-shanda/1-feature/9-Final_Segmented_Genes`
- `2-8.3-shanda/1-feature/9-Summary_Results`

### `figure/figure4/figure-4-1-human-gene-ratio.ipynb`

- `2-8.3-shanda/1-feature/1-human-guaidian-choose-gene/Gene_Lists`

### `figure/figure5/figure-5-1-plasma.ipynb`

- `0-figure-code`
- `2-8.3-shanda/1-feature/1-figure/0-result-6/VTN_stable_high_receptor_celltypes`

### `figure/figure6/figure-6-0-clock.ipynb`

- `1-TMS-remove/2-restart`
- `2-8.3-shanda/1-data/2-valid/GSE137869_RAW`
- `2-8.3-shanda/1-feature/9-Final_Segmented_Genes`
- `2-8.3-shanda/1-feature/mouse_rat_orthologs.csv`
- `header.txt`

### `figure/figure6/figure-6-1-leipameisu.ipynb`

- `2-8.3-shanda/1-data/2-valid/GSE210669_RAW`
- `2-8.3-shanda/1-feature/10-2-0-Gene_Trajectory_Clusters_Annotated/{TARGET_TISSUE}_Cluster_Assignments.csv`
- `2-8.3-shanda/1-feature/9-Master-raw-Master_Clocks_3/{TARGET_TISSUE}_Clock.pkl`

### `figure/figure6/figure-6-2-CR.ipynb`

- `2-8.3-shanda/1-data/2-valid/GSE137869_RAW`
- `2-8.3-shanda/1-feature/9-Master-raw-Master_Clocks_3`

### `figure/figure6/figure-6-3-target-gene.ipynb`

- `2-8.3-shanda/1-feature/1-figure/0-4-result-5-CR-leipameisu/2-CR/mmc2.xlsx`
- `2-8.3-shanda/1-feature/1-figure/0-4-result-5-CR-leipameisu/3-CR/table.xlsx`

## Inputs that need explicit release information

- `figure/data- preprocess.ipynb` references Tabula Sapiens `.h5ad` data and writes derived HDF5, metadata, and gene-list files. The original dataset accession and preprocessing decisions need author confirmation.
- Figure 2 MLP comparisons use selected-gene lists, train/test HDF5 files, competitor model results, and summary tables. The bundled Heart example alone does not supply these.
- Figure 3/4 analyses use gene identifiers, tissue-level expression matrices, trajectory clusters, and enrichment results. Their biological labels and source repositories need documentation.
- The intervention and clock analyses use external 10x inputs, precomputed clock `.pkl` models, cluster assignments, ortholog tables, and Excel sheets. Release or accession details and model provenance are pending.
- `figure/figure5/figure-5-1-plasma.ipynb` reads cached CellPhoneDB CSVs from `0-figure-code/0-result-6-2-plasma-data-output/22_cellphonedb_tissue_pseudobulk/24_age_stratified_lt60_gt60/`, plus plasma VTN and receptor-expression tables. Its helper module is now included, but these result tables and the source data behind them are external.
- `figure/figure5/cellphonedb_interaction_count_network.py` requires `22_cpdb_all_pvalues.csv`, `22_cpdb_all_means.csv`, `22_eight_ligand_all_cpdb_receptor_candidates.csv`, and `04b_tissue_secreted_gene_plasma_evidence.csv` under the listed input locations.

For submission, connect each reported panel to an accessible dataset or source-data table, the exact input path, and the command or notebook cell sequence. Record restrictions and an access route for data that cannot be publicly redistributed.
