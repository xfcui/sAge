from __future__ import annotations

import os
from pathlib import Path

from ..paths import input_path, output_path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle, Patch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd


INPUT_DIR = Path(input_path("0-figure-code/0-result-6-2-plasma-data-output"))
WORK_INPUT_DIR = INPUT_DIR / "22_cellphonedb_tissue_pseudobulk"
WORK_DIR = Path(output_path("figure5/plasma/cellphonedb_network"))
PVALUES_FILE = WORK_INPUT_DIR / "22_cpdb_all_pvalues.csv"
MEANS_FILE = WORK_INPUT_DIR / "22_cpdb_all_means.csv"
CANDIDATE_FILE = WORK_INPUT_DIR / "22_eight_ligand_all_cpdb_receptor_candidates.csv"
SOURCE_FILE = INPUT_DIR / "04b_tissue_secreted_gene_plasma_evidence.csv"

P_THRESHOLD = float(os.environ.get("CPDB_ROUTE_P_THRESHOLD", "0.001"))
FDR_THRESHOLD = float(os.environ.get("CPDB_ROUTE_FDR_THRESHOLD", "0.05"))
N_PERMUTATIONS = int(os.environ.get("CPDB_ROUTE_N_PERMUTATIONS", "1000"))
if np.isclose(P_THRESHOLD, 0.001):
    P_TAG = "p0001"  # retained for backward-compatible output names
elif np.isclose(P_THRESHOLD, 0.01):
    P_TAG = "p001"
elif np.isclose(P_THRESHOLD, 0.05):
    P_TAG = "p005"
else:
    P_TAG = "p" + str(P_THRESHOLD).replace("0.", "0").replace(".", "_")
FDR_TAG = "fdr" + str(FDR_THRESHOLD).replace("0.", "0").replace(".", "_")
ANALYSIS_TAG = f"{P_TAG}_{FDR_TAG}"
FOCAL_LIGANDS = ["LYZ", "CST3", "VEGFA", "CTSB", "IGFBP6", "VTN", "LCN2", "SELENOP"]

HYPOTHESIS_FILE = WORK_DIR / f"22_cpdb_all_candidate_routes_scRNA_{ANALYSIS_TAG}.csv"
ROUTE_FILE = WORK_DIR / f"22_cpdb_significant_routes_scRNA_{ANALYSIS_TAG}.csv"
EDGE_FILE = WORK_DIR / f"22_cpdb_tissue_lr_pair_counts_scRNA_{ANALYSIS_TAG}.csv"
PAIR_FILE = WORK_DIR / f"22_cpdb_significant_pair_summary_scRNA_{ANALYSIS_TAG}.csv"
LIGAND_FILE = WORK_DIR / f"22_cpdb_eight_ligand_test_summary_scRNA_{ANALYSIS_TAG}.csv"
FIGURE_STEM = WORK_DIR / f"22_cpdb_tissue_lr_pair_count_network_scRNA_{ANALYSIS_TAG}_journal"
PAIR_FIGURE_DIR = WORK_DIR / f"22_cpdb_individual_pair_networks_scRNA_{ANALYSIS_TAG}_journal"
ALLUVIAL_STEM = WORK_DIR / f"23_all_source_plasma_tissue_sankey_scRNA_{ANALYSIS_TAG}"
ALL_PAIR_DOT_STEM = WORK_DIR / f"23_all_significant_lr_pair_dotplot_scRNA_{ANALYSIS_TAG}"
ALL_PAIR_MATRIX_FILE = WORK_DIR / f"23_all_significant_lr_pair_tissue_matrix_scRNA_{ANALYSIS_TAG}.csv"


def key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def clean_pair(row: pd.Series) -> str:
    members = str(row.get("receptor_components", "")).split(";")
    # Conventional integrin notation places alpha before beta (ITGAV+ITGB1).
    members = sorted(members, key=lambda x: (0 if x.startswith("ITGA") else 1, x))
    receptor = "+".join(members)
    return f"{row['ligand_gene']} → {receptor}"


def partner_key(value: str) -> str:
    value = str(value)
    return value.split(":", 1)[1] if ":" in value else value


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """BH-adjust a prespecified family while preserving index and missing values."""
    numeric = pd.to_numeric(pvalues, errors="coerce")
    valid = numeric.notna()
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    if not valid.any():
        return adjusted
    values = numeric.loc[valid].to_numpy(dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    n_tests = len(ranked)
    ranked_adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    restored = np.empty(n_tests, dtype=float)
    restored[order] = np.clip(ranked_adjusted, 0.0, 1.0)
    adjusted.loc[valid] = restored
    return adjusted


def finite_permutation_pvalue(pvalues: pd.Series, n_permutations: int) -> pd.Series:
    """Replace Monte Carlo zero p-values using (b + 1) / (B + 1)."""
    numeric = pd.to_numeric(pvalues, errors="coerce")
    exceedances = np.rint(numeric * n_permutations)
    return ((exceedances + 1.0) / (n_permutations + 1.0)).where(numeric.notna())


def build_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pvalues = pd.read_csv(PVALUES_FILE)
    means = pd.read_csv(MEANS_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)
    source_evidence = pd.read_csv(SOURCE_FILE)
    for table in (pvalues, means, candidates):
        table["partner_a_key"] = table["partner_a"].map(partner_key)
        table["partner_b_key"] = table["partner_b"].map(partner_key)

    source_map = (
        source_evidence.loc[source_evidence["gene_symbol"].isin(FOCAL_LIGANDS), ["gene_symbol", "tissue"]]
        .drop_duplicates()
        .assign(source_tissue=lambda x: x["tissue"].map(key))
        .groupby("gene_symbol")["source_tissue"].agg(lambda x: sorted(set(x))).to_dict()
    )
    pair_columns = [column for column in pvalues.columns if "|" in column]
    meta_columns = [column for column in pvalues.columns if column not in pair_columns]

    pv_long = pvalues.melt(
        id_vars=meta_columns, value_vars=pair_columns,
        var_name="tissue_pair", value_name="cellphonedb_p",
    )
    mean_meta = [column for column in means.columns if column not in pair_columns]
    mean_long = means.melt(
        id_vars=mean_meta, value_vars=pair_columns,
        var_name="tissue_pair", value_name="cellphonedb_mean",
    )[["id_cp_interaction", "tissue_pair", "cellphonedb_mean"]]
    long = pv_long.merge(mean_long, on=["id_cp_interaction", "tissue_pair"], how="left")
    long[["source_tissue", "target_tissue"]] = long["tissue_pair"].str.split("|", n=1, expand=True)
    # The distributed input CSV does not carry CellPhoneDB's generated CPI ids.
    # partner_a/partner_b are stable on both sides and therefore form the audit key.
    long = long.merge(
        candidates[["partner_a_key", "partner_b_key", "ligand_gene", "ligand_components", "receptor_components"]],
        on=["partner_a_key", "partner_b_key"], how="inner", validate="many_to_one",
    )
    long["source_is_sage_context"] = long.apply(
        lambda r: r["source_tissue"] in source_map.get(r["ligand_gene"], []), axis=1
    )
    long["ligand_receptor_pair"] = long.apply(clean_pair, axis=1)

    # The multiplicity family is defined before inspecting significance: every
    # prespecified SAGE-source ligand–receptor combination across all 24 targets.
    candidate_family = long.loc[long["source_is_sage_context"]].copy()
    candidate_family["cellphonedb_p"] = pd.to_numeric(
        candidate_family["cellphonedb_p"], errors="coerce"
    )
    candidate_family["cellphonedb_p_permutation_corrected"] = finite_permutation_pvalue(
        candidate_family["cellphonedb_p"], N_PERMUTATIONS
    )
    candidate_family["cellphonedb_fdr_bh"] = benjamini_hochberg(
        candidate_family["cellphonedb_p_permutation_corrected"]
    )
    candidate_family["passes_raw_p"] = (
        candidate_family["cellphonedb_p_permutation_corrected"] < P_THRESHOLD
    )
    candidate_family["passes_fdr"] = candidate_family["cellphonedb_fdr_bh"] < FDR_THRESHOLD
    candidate_family["significant"] = (
        candidate_family["passes_raw_p"] & candidate_family["passes_fdr"]
    )
    candidate_family.to_csv(HYPOTHESIS_FILE, index=False, encoding="utf-8-sig")

    routes = candidate_family.loc[candidate_family["significant"]].copy()
    routes = routes.sort_values(
        ["source_tissue", "target_tissue", "ligand_receptor_pair"]
    ).reset_index(drop=True)
    keep = [
        "ligand_gene", "ligand_receptor_pair", "receptor_components",
        "source_tissue", "target_tissue", "cellphonedb_p",
        "cellphonedb_p_permutation_corrected", "cellphonedb_fdr_bh",
        "cellphonedb_mean",
        "id_cp_interaction", "interacting_pair",
    ]
    routes[keep].to_csv(ROUTE_FILE, index=False, encoding="utf-8-sig")

    edges = (
        routes.groupby(["source_tissue", "target_tissue"], as_index=False)
        .agg(
            # Edge width is defined by the number of distinct significant
            # ligand–receptor gene pairs, not by raw table rows or route score.
            lr_pair_count=("ligand_receptor_pair", "nunique"),
            cpdb_interaction_id_count=("id_cp_interaction", "nunique"),
            significant_pairs=("ligand_receptor_pair", lambda x: "; ".join(sorted(set(x)))),
            min_cellphonedb_p=("cellphonedb_p", "min"),
            max_corrected_permutation_p=("cellphonedb_p_permutation_corrected", "max"),
            max_cellphonedb_fdr_bh=("cellphonedb_fdr_bh", "max"),
        )
        .sort_values(["source_tissue", "target_tissue"])
    )
    listed_pair_count = edges["significant_pairs"].str.split("; ").map(len)
    if not edges["lr_pair_count"].eq(listed_pair_count).all():
        raise ValueError("LR pair count does not match the exported distinct pair list.")
    # This audit field may differ in other CellPhoneDB databases when several
    # interaction IDs map to the same gene-level LR pair; only lr_pair_count is
    # used for the visual encoding requested here.
    edges["edge_width_points"] = edges["lr_pair_count"].map(edge_width_from_lr_count)
    edges.to_csv(EDGE_FILE, index=False, encoding="utf-8-sig")

    pair_summary = (
        routes.groupby(["ligand_gene", "ligand_receptor_pair", "receptor_components"], as_index=False)
        .agg(
            n_significant_routes=("target_tissue", "size"),
            n_target_tissues=("target_tissue", "nunique"),
            target_tissues=("target_tissue", lambda x: "; ".join(sorted(set(x)))),
        )
        .sort_values(["ligand_gene", "ligand_receptor_pair"])
    )
    pair_summary.to_csv(PAIR_FILE, index=False, encoding="utf-8-sig")

    coverage = []
    for ligand in FOCAL_LIGANDS:
        cpdb_ids = candidates.loc[candidates["ligand_gene"].eq(ligand), "interacting_pair"].nunique()
        focal = routes.loc[routes["ligand_gene"].eq(ligand)]
        coverage.append({
            "ligand_gene": ligand,
            "sage_source_tissues": "; ".join(source_map.get(ligand, [])),
            "n_cpdb_candidate_pairs": int(cpdb_ids),
            "n_significant_pairs": int(focal["id_cp_interaction"].nunique()),
            "n_significant_tissue_routes": int(len(focal)),
            "n_target_tissues": int(focal["target_tissue"].nunique()),
        })
    ligand_summary = pd.DataFrame(coverage)
    ligand_summary.to_csv(LIGAND_FILE, index=False, encoding="utf-8-sig")
    return routes, edges, pair_summary, ligand_summary


# Fixed, muted categorical palette: node color denotes tissue identity only.
# Direct labels, dark keylines and a white halo preserve legibility in print.
TISSUE_COLORS = {
    "bladder": "#E07B91",
    "blood": "#A75893",
    "bone_marrow": "#877A99",
    "eye": "#E58B35",
    "fat": "#C5A12D",
    "heart": "#D75B57",
    "kidney": "#7566A6",
    "large_intestine": "#B77B4C",
    "liver": "#78A641",
    "lung": "#32AD88",
    "lymph_node": "#9465A8",
    "mammary": "#CB79A5",
    "muscle": "#617D99",
    "pancreas": "#3AA5C1",
    "prostate": "#74828F",
    "salivary_gland": "#4B9C98",
    "skin": "#4F86C6",
    "small_intestine": "#D09A62",
    "spleen": "#8059A0",
    "thymus": "#6877B5",
    "tongue": "#A56B5C",
    "trachea": "#B45DB5",
    "uterus": "#E06C74",
    "vasculature": "#D84BA5",
}

# Explicit hue-wheel order for Figure 4H receiver stacks.  The sequence moves
# from red through orange, yellow, green, cyan, blue and purple; nearby muted
# pink/brown/grey categories are placed beside their closest visual family.
FIG4H_RAINBOW_TISSUE_ORDER = [
    "vasculature", "trachea", "blood", "lymph_node", "spleen",
    "prostate", "bone_marrow", "kidney", "thymus", "muscle", "skin",
    "pancreas", "salivary_gland", "lung",
    "liver", "fat",
    "tongue", "small_intestine", "large_intestine", "eye",
    "mammary", "bladder", "uterus", "heart",
]


def node_positions(nodes: list[str], source: str) -> dict[str, np.ndarray]:
    if len(nodes) > 16:
        # Dense network: a true circular layout keeps all 24 labels separated.
        anchors = {"liver": 0.0, "fat": 120.0, "salivary_gland": 240.0}
        pos = {
            node: 0.93 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            for node, angle in anchors.items() if node in nodes
        }
        others = sorted(node for node in nodes if node not in anchors)
        open_angles = np.concatenate([
            np.linspace(15, 105, 7), np.linspace(135, 225, 7), np.linspace(255, 345, 7)
        ])[:len(others)]
        for node, angle in zip(others, open_angles):
            pos[node] = 0.93 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        return pos
    targets = sorted(node for node in nodes if node != source)
    # Put the source at the right and fan receiving tissues around the left arc.
    pos = {source: np.array([0.74, 0.0])}
    angles = np.linspace(np.deg2rad(65), np.deg2rad(295), len(targets))
    for node, angle in zip(targets, angles):
        pos[node] = np.array([0.93 * np.cos(angle) - 0.10, 0.93 * np.sin(angle)])
    return pos


def flow_values(edges: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in edges.itertuples(index=False):
        n = float(row.lr_pair_count)
        values[row.source_tissue] = values.get(row.source_tissue, 0.0) + n
        values[row.target_tissue] = values.get(row.target_tissue, 0.0) + n
    return values


def edge_width_from_lr_count(lr_pair_count: float) -> float:
    """Linear, auditable mapping from distinct LR-pair count to line width."""
    return 0.55 + 0.70 * float(lr_pair_count)


LR_PAIR_COLORS = {
    "VTN → ITGA2B+ITGB3": "#9B4D62",
    "VTN → ITGAV+ITGB1": "#C87355",
    "VTN → ITGAV+ITGB3": "#D7A04B",
    "LCN2 → SLC22A17": "#318A8A",
}


def draw_network(
    edges: pd.DataFrame,
    title: str,
    stem: Path,
    pair_label: str | None = None,
    routes: pd.DataFrame | None = None,
    node_universe: list[str] | None = None,
    positions_override: dict[str, np.ndarray] | None = None,
    max_count_override: float | None = None,
    max_flow_override: float | None = None,
    subtitle_note: str | None = None,
    label_edge_counts: bool = True,
    flow_legend_examples: list[float] | None = None,
) -> None:
    if edges.empty:
        return
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    observed_nodes = set(edges["source_tissue"]) | set(edges["target_tissue"])
    nodes = sorted(set(node_universe) | observed_nodes) if node_universe else sorted(observed_nodes)
    dense_network = len(nodes) > 16
    source = edges["source_tissue"].mode().iloc[0]
    show_pair_strands = dense_network and routes is not None and pair_label is not None
    if positions_override is not None:
        missing_positions = sorted(set(nodes) - set(positions_override))
        if missing_positions:
            raise ValueError(f"Missing fixed positions for nodes: {missing_positions}")
        pos = {node: np.asarray(positions_override[node], dtype=float) for node in nodes}
    elif dense_network:
        # Place every tissue on the circumference.  Rotate the alphabetical
        # order so that the dominant source (liver) sits at three o'clock, but
        # remains a peripheral node rather than a visual centre.
        circular_nodes = sorted(nodes)
        if "liver" in circular_nodes:
            liver_index = circular_nodes.index("liver")
            circular_nodes = circular_nodes[liver_index:] + circular_nodes[:liver_index]
        angles = np.linspace(0.0, 2.0 * np.pi, len(circular_nodes), endpoint=False)
        pos = {
            node: np.array([0.96 * np.cos(angle), 0.96 * np.sin(angle)])
            for node, angle in zip(circular_nodes, angles)
        }
    else:
        pos = node_positions(nodes, source)
    flows = flow_values(edges)
    for node in nodes:
        flows.setdefault(node, 0.0)
    counts = edges["lr_pair_count"].astype(float)
    max_count = max(float(max_count_override or counts.max()), 1.0)
    # Match the reference figure's sequential interaction-count encoding:
    # low counts are pale blush and high counts are saturated magenta-red.
    cmap = LinearSegmentedColormap.from_list(
        "interaction_count", ["#F8E2E7", "#F0B0BD", "#E47B91"]
    )
    norm = Normalize(vmin=0, vmax=max_count)

    fig = plt.figure(figsize=(9.4, 7.5), facecolor="white")
    ax = fig.add_axes([0.045, 0.17, 0.70, 0.71])
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw only a very light structural guide beneath the colored LR strands.
    for edge_index, row in enumerate(edges.itertuples(index=False)):
        start, end = pos[row.source_tissue], pos[row.target_tissue]
        count = float(row.lr_pair_count)
        width = edge_width_from_lr_count(count)
        color = "#D9DEE3" if show_pair_strands else cmap(norm(count))
        if row.source_tissue == row.target_tissue:
            radial = start / max(np.linalg.norm(start), 1e-8)
            tangent = np.array([-radial[1], radial[0]])
            loop_start = start + 0.035 * radial + 0.075 * tangent
            loop_end = start + 0.035 * radial - 0.075 * tangent
            patch = FancyArrowPatch(
                loop_start, loop_end,
                connectionstyle="arc3,rad=-2.25",
                arrowstyle="-" if show_pair_strands else "-|>",
                mutation_scale=12,
                linewidth=max(0.55, width * 0.72) if show_pair_strands else width,
                color=color, alpha=0.28 if show_pair_strands else 0.90,
                shrinkA=4, shrinkB=4,
            )
        else:
            # In the circular tissue network, straight chords from the common
            # source cannot cross one another (apart from their shared source
            # endpoint).  Alternating curvature created avoidable crossings.
            rad = 0.0 if dense_network else 0.075 * (1 if edge_index % 2 == 0 else -1)
            patch = FancyArrowPatch(
                start, end, connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-" if show_pair_strands else "-|>",
                mutation_scale=9.0,
                linewidth=max(0.55, width * 0.72) if show_pair_strands else width,
                color=color, alpha=0.25 if show_pair_strands else 0.88,
                shrinkA=15, shrinkB=15,
            )
        ax.add_patch(patch)
        if pair_label is None and not dense_network and label_edge_counts:
            mid = start + np.array([0.28, 0.0]) if row.source_tissue == row.target_tissue else (start + end) / 2
            ax.text(
                mid[0], mid[1], str(int(count)), ha="center", va="center", fontsize=7.5,
                color="#6C3034", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88),
                zorder=8,
            )

    if show_pair_strands:
        for (_, _), sub in routes.groupby(["source_tissue", "target_tissue"], sort=True):
            sub = sub.sort_values("ligand_receptor_pair")
            offsets = np.linspace(-0.020, 0.020, len(sub)) if len(sub) > 1 else [0.0]
            for offset, row in zip(offsets, sub.itertuples(index=False)):
                start, end = pos[row.source_tissue], pos[row.target_tissue]
                strand_color = LR_PAIR_COLORS.get(row.ligand_receptor_pair, "#607080")
                if row.source_tissue == row.target_tissue:
                    loop_start = start + np.array([0.055, 0.055 + offset])
                    loop_end = start + np.array([0.055, -0.055 + offset])
                    patch = FancyArrowPatch(
                        loop_start, loop_end, connectionstyle=f"arc3,rad={-2.0 - offset * 8}",
                        arrowstyle="-|>", mutation_scale=7.5, linewidth=0.85,
                        color=strand_color, alpha=0.70, shrinkA=3, shrinkB=3, zorder=6,
                    )
                else:
                    patch = FancyArrowPatch(
                        start, end, connectionstyle=f"arc3,rad={offset}", arrowstyle="-|>",
                        mutation_scale=7.2, linewidth=0.80, color=strand_color,
                        alpha=0.68, shrinkA=15, shrinkB=15, zorder=6,
                    )
                ax.add_patch(patch)

    max_flow = max(float(max_flow_override or max(flows.values())), 1.0)
    self_nodes = set(edges.loc[edges["source_tissue"].eq(edges["target_tissue"]), "source_tissue"])
    for node in nodes:
        xy = pos[node]
        size = 125 + 470 * np.sqrt(flows[node] / max_flow)
        # White outer halo separates nodes from crossing edges; the fine dark
        # keyline keeps pale fills legible in print and at reduced size.
        ax.scatter(*xy, s=size + 70, color="white", edgecolor="white", linewidth=1.0, zorder=9)
        ax.scatter(
            *xy, s=size, color=TISSUE_COLORS[node],
            edgecolor="#26313C", linewidth=0.85, zorder=10,
        )
        if dense_network and node == "liver":
            # Keep the source label outside its node, using the same typography
            # as every other tissue, while avoiding the external self-loop.
            offset = np.array([0.0, -0.135])
            ha = "center"
        elif dense_network:
            radial = xy / max(np.linalg.norm(xy), 1e-8)
            offset = radial * 0.115
            ha = "left" if radial[0] > 0.18 else ("right" if radial[0] < -0.18 else "center")
        else:
            offset = np.array([0.0, 0.10]) if node != source else np.array([0.0, -0.13])
            ha = "center"
        ax.text(
            *(xy + offset), node.replace("_", " ").capitalize(), ha=ha, va="center",
            color="#202832",
            fontsize=8.45 if dense_network else 8.8,
            fontweight="medium", bbox=None, zorder=12,
        )

    ax.set_xlim(-1.30 if dense_network else -1.28, 1.30 if dense_network else 1.32)
    ax.set_ylim(-1.25, 1.25)
    fig.text(0.055, 0.947, title, fontsize=15.5, fontweight="bold", color="#17253A")
    if pair_label:
        subtitle = f"CellPhoneDB P < {P_THRESHOLD:g} and BH-FDR < {FDR_THRESHOLD:g}"
    else:
        subtitle = (
            f"CellPhoneDB P < {P_THRESHOLD:g} and BH-FDR < {FDR_THRESHOLD:g}  |  "
            "edge width = distinct significant ligand–receptor pair count"
        )
    fig.text(0.055, 0.914, subtitle, fontsize=9.2, color="#667383")
    fig.text(
        0.055, 0.888,
        "Node fill = tissue identity  |  node area = total significant LR-pair flow",
        fontsize=8.3, color="#7A8490",
    )
    if subtitle_note:
        fig.text(0.055, 0.864, subtitle_note, fontsize=7.8, color="#7A8490")
    if pair_label:
        pair_label_y = 0.842 if subtitle_note else 0.864
        fig.text(0.055, pair_label_y, pair_label, fontsize=10.2,
                 color="#873C43", fontweight="bold")

    legend_ax = fig.add_axes([0.765, 0.17, 0.21, 0.66])
    legend_ax.axis("off")
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    if pair_label is None:
        # Separate continuous color and width keys, matching the reference
        # figure rather than annotating individual colored lines with counts.
        colorbar_ax = fig.add_axes([0.785, 0.665, 0.018, 0.145])
        scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        colorbar = fig.colorbar(scalar_map, cax=colorbar_ax, orientation="vertical")
        colorbar.set_ticks(np.arange(0, int(max_count) + 1))
        colorbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.6, colors="#202832")
        colorbar.set_label("Interaction count", fontsize=8.2, color="#202832", labelpad=7)
        colorbar.outline.set_visible(False)

        legend_ax.text(
            0.00, 0.52, "Interaction count", fontsize=8.5, color="#202832",
            rotation=90, ha="center", va="center",
        )
        width_counts = list(range(0, int(max_count) + 1))
        for idx, count in enumerate(width_counts):
            y = 0.60 - idx * 0.075
            legend_ax.plot(
                [0.12, 0.28], [y, y], lw=edge_width_from_lr_count(count),
                color="#111111", solid_capstyle="butt",
            )
            legend_ax.text(0.36, y, f"{count}", va="center", fontsize=8.0, color="#202832")
        y0 = 0.30
    else:
        legend_ax.text(0.0, 0.96, "Interaction count", fontsize=10, fontweight="bold", color="#17253A")
        legend_ax.plot(
            [0.02, 0.35], [0.87, 0.87], lw=edge_width_from_lr_count(1),
            color=cmap(norm(1)), solid_capstyle="round",
        )
        legend_ax.text(0.43, 0.87, "1", va="center", fontsize=8.5)
        y0 = 0.48
    legend_ax.text(0.0, y0, "Node area", fontsize=10, fontweight="bold", color="#17253A")
    legend_ax.text(0.0, y0 - 0.05, "Total significant LR-pair flow", fontsize=7.8, color="#667383")
    examples = (
        sorted(set(float(value) for value in flow_legend_examples))
        if flow_legend_examples is not None
        else sorted(set([min(flows.values()), max(flows.values())]))
    )
    for idx, value in enumerate(examples):
        y = y0 - 0.14 - idx * 0.14
        size = 125 + 470 * np.sqrt(value / max_flow)
        legend_ax.scatter(
            0.14, y, s=size, facecolor="#E7EEF2", edgecolor="#30343B",
            linewidth=0.7, clip_on=False,
        )
        legend_ax.text(0.43, y, f"{value:g}", va="center", fontsize=8.5)
    fig.text(0.765, 0.115, "Arrow: source tissue → potential receiver", fontsize=7.7, color="#667383")

    for extension in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=600 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(routes: pd.DataFrame, edges: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    draw_network(
        edges,
        "Plasma-supported inter-tissue aging communication",
        FIGURE_STEM,
        routes=routes,
    )
    PAIR_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for pair in pair_summary["ligand_receptor_pair"]:
        sub = routes.loc[routes["ligand_receptor_pair"].eq(pair)]
        pair_edges = (
            sub.groupby(["source_tissue", "target_tissue"], as_index=False)
            .agg(
                lr_pair_count=("ligand_receptor_pair", "nunique"),
                cpdb_interaction_id_count=("id_cp_interaction", "nunique"),
            )
        )
        safe = pair.replace(" ", "_").replace("→", "to").replace("+", "_")
        draw_network(
            pair_edges,
            "Significant inter-tissue communication route",
            PAIR_FIGURE_DIR / safe,
            pair_label=pair,
            routes=sub,
        )


def _save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=600 if extension == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def _ribbon(ax: plt.Axes, x0: float, x1: float, y0a: float, y0b: float,
            y1a: float, y1b: float, color: str, alpha: float = 0.55) -> None:
    control = (x1 - x0) * 0.48
    vertices = [
        (x0, y0a), (x0 + control, y0a), (x1 - control, y1a), (x1, y1a),
        (x1, y1b), (x1 - control, y1b), (x0 + control, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color,
                           edgecolor="white", linewidth=0.22,
                           alpha=alpha, zorder=1))


def draw_fig4h_alluvial(routes: pd.DataFrame) -> None:
    """Figure-4H-style Sankey containing every retained ligand-source tissue."""
    data = routes.copy()
    if data.empty:
        return
    source_counts = data.groupby("source_tissue").size().sort_values()
    ligand_counts = data.groupby("ligand_gene").size().sort_values()
    target_counts_raw = data.groupby("target_tissue").size()
    target_order = [t for t in FIG4H_RAINBOW_TISSUE_ORDER if t in target_counts_raw.index]
    target_order.extend(sorted(set(target_counts_raw.index) - set(target_order)))
    target_counts = target_counts_raw.reindex(target_order)
    total = int(len(data))
    fig = plt.figure(figsize=(8.8, 5.1), facecolor="white")
    ax = fig.add_axes([0.09, 0.18, 0.51, 0.64])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_ylabel("Number of interaction pairs", fontsize=10)
    ax.set_yticks(np.unique(np.rint(np.linspace(0, total, 4)).astype(int)))
    ax.tick_params(axis="y", labelsize=8, width=0.7, length=3)

    x_source, x_plasma, x_target, bar_width = 0.16, 0.47, 0.78, 0.075
    plasma_color = "#F47F82"

    # Left bar: ligand-expressing source tissues.
    source_starts: dict[str, float] = {}
    cursor = 0.0
    for tissue, count in source_counts.items():
        source_starts[tissue] = cursor
        ax.add_patch(Rectangle((x_source, cursor), bar_width, count,
                               facecolor=TISSUE_COLORS[tissue], edgecolor="#30343B",
                               linewidth=0.7, zorder=3))
        cursor += float(count)

    # Middle plasma bar; internal coordinate blocks are assigned by ligand.
    ax.add_patch(Rectangle((x_plasma, 0), bar_width, total, facecolor=plasma_color,
                           edgecolor="#30343B", linewidth=0.8, zorder=3))
    ligand_starts: dict[str, float] = {}
    cursor = 0.0
    for ligand, count in ligand_counts.items():
        ligand_starts[ligand] = cursor
        cursor += float(count)

    # Source tissue → plasma ligand.  Aggregate identical source/ligand routes,
    # as in the compact reference Sankey.
    source_offsets = {key: 0.0 for key in source_starts}
    ligand_in_offsets = {key: 0.0 for key in ligand_starts}
    source_ligand = data.groupby(["source_tissue", "ligand_gene"]).size().reset_index(name="count")
    for row in source_ligand.itertuples(index=False):
        left0 = source_starts[row.source_tissue] + source_offsets[row.source_tissue]
        mid0 = ligand_starts[row.ligand_gene] + ligand_in_offsets[row.ligand_gene]
        _ribbon(ax, x_source + bar_width, x_plasma, left0, left0 + row.count,
                mid0, mid0 + row.count, color="#F2A5AC", alpha=0.60)
        source_offsets[row.source_tissue] += row.count
        ligand_in_offsets[row.ligand_gene] += row.count

    # Receiver bar coordinates are defined by the number of distinct
    # significant LR pairs terminating in each tissue.
    target_starts: dict[str, float] = {}
    cursor = 0.0
    for tissue, count in target_counts.items():
        target_starts[tissue] = cursor
        cursor += float(count)

    # Plasma → receiver: one ribbon per receiving tissue.  Ribbon width is the
    # number of distinct significant LR pairs, not one strand per pair.  The
    # plasma-side order is data-driven (count, then summed CPDB mean), while the
    # receiver bar remains alphabetical; this retains an alluvial read without
    # exposing individual LR-pair strands.
    receiver_summary = (
        data.groupby("target_tissue", as_index=False)
        .agg(
            lr_pair_count=("id_cp_interaction", "nunique"),
            summed_interaction=("cellphonedb_mean", "sum"),
        )
        .sort_values(["lr_pair_count", "summed_interaction", "target_tissue"],
                     ascending=[False, False, True])
    )
    plasma_cursor = 0.0
    for row in receiver_summary.itertuples(index=False):
        width = float(row.lr_pair_count)
        receiver_start = target_starts[row.target_tissue]
        _ribbon(ax, x_plasma + bar_width, x_target,
                plasma_cursor, plasma_cursor + width,
                receiver_start, receiver_start + width,
                color="#F2A5B1", alpha=0.58)
        plasma_cursor += width

    cursor = 0.0
    for tissue, count in target_counts.items():
        ax.add_patch(Rectangle((x_target, cursor), bar_width, count,
                               facecolor=TISSUE_COLORS[tissue], edgecolor="#30343B",
                               linewidth=0.42, zorder=3))
        cursor += float(count)

    label_offset = max(0.05 * total, 0.25)
    arrow_offset = max(0.105 * total, 0.55)
    bottom_label_offset = max(0.07 * total, 0.35)
    bottom_arrow_offset = max(0.11 * total, 0.65)
    for x, top, bottom in [
        (x_source + bar_width / 2, "Ligand", "Tissue"),
        (x_plasma + bar_width / 2, "Ligand", "Plasma"),
        (x_target + bar_width / 2, "Receptor", "Tissue"),
    ]:
        ax.text(x, total + label_offset, top, ha="center", va="bottom", fontsize=10)
        ax.text(x, -bottom_label_offset, bottom, ha="center", va="top", fontsize=9)

    ax.annotate("", xy=(x_plasma + 0.01, total + arrow_offset),
                xytext=(x_source + bar_width, total + arrow_offset),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.20",
                                color="#111111", lw=0.8), annotation_clip=False)
    ax.annotate("", xy=(x_target + 0.01, total + arrow_offset),
                xytext=(x_plasma + bar_width, total + arrow_offset),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.20",
                                color="#111111", lw=0.8), annotation_clip=False)
    ax.annotate("", xy=(x_plasma + 0.01, -bottom_arrow_offset),
                xytext=(x_source + bar_width, -bottom_arrow_offset),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.20",
                                color="#111111", lw=0.8), annotation_clip=False)
    ax.annotate("", xy=(x_target + 0.01, -bottom_arrow_offset),
                xytext=(x_plasma + bar_width, -bottom_arrow_offset),
                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.20",
                                color="#111111", lw=0.8), annotation_clip=False)

    legend_ax = fig.add_axes([0.63, 0.19, 0.35, 0.62])
    legend_ax.axis("off")
    legend_ax.text(0.0, 1.02, "Tissue", fontsize=10, fontweight="bold",
                   color="#17253A", transform=legend_ax.transAxes)
    # Legend reads top-to-bottom, matching the visual order of the stacked bar.
    tissue_names = list(reversed(target_order)) + [
        t for t in source_counts.index if t not in target_order
    ]
    handles = [Patch(facecolor=TISSUE_COLORS[t], edgecolor="#30343B", linewidth=0.4,
                     label=t.replace("_", " ").capitalize()) for t in tissue_names]
    handles.append(Patch(facecolor=plasma_color, edgecolor="#30343B", linewidth=0.4,
                         label="Plasma"))
    legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0.98),
                     frameon=False, ncol=3, fontsize=6.8, handlelength=0.9,
                     handleheight=0.9, columnspacing=0.75, labelspacing=0.55)
    legend_ax.text(0.0, 0.17, "Interaction count", fontsize=9.2,
                   color="#17253A", transform=legend_ax.transAxes)
    legend_ax.plot([0.02, 0.22], [0.11, 0.11], color="#F2A5B1", lw=3.0,
                   transform=legend_ax.transAxes, solid_capstyle="butt")
    legend_ax.text(0.27, 0.11, "Ribbon width = LR-pair count", fontsize=8, va="center",
                   transform=legend_ax.transAxes)
    fig.text(0.035, 0.94, "H", fontsize=18, fontweight="bold")
    _save_publication_figure(fig, ALLUVIAL_STEM)


def draw_all_pair_fig4i_dotplot(routes: pd.DataFrame) -> None:
    """Figure-4I-inspired matrix of every significant LR route in the analysis."""
    matrix = routes.copy()
    if matrix.empty:
        return
    matrix["route_label"] = (
        matrix["source_tissue"].str.replace("_", " ").str.capitalize()
        + " : "
        + matrix["target_tissue"].str.replace("_", " ").str.capitalize()
    )
    route_order = (
        matrix.groupby("route_label")["cellphonedb_mean"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    pair_order = (
        matrix.groupby("ligand_receptor_pair")["cellphonedb_mean"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    matrix["route_label"] = pd.Categorical(matrix["route_label"], route_order, ordered=True)
    matrix["ligand_receptor_pair"] = pd.Categorical(
        matrix["ligand_receptor_pair"], pair_order, ordered=True
    )
    matrix = matrix.sort_values(["ligand_receptor_pair", "route_label"])
    matrix.to_csv(ALL_PAIR_MATRIX_FILE, index=False)

    values = matrix["cellphonedb_mean"].astype(float)
    vmin, vmax = float(values.min()), float(values.max())
    norm_mean = Normalize(vmin=vmin, vmax=vmax)
    cmap_mean = LinearSegmentedColormap.from_list(
        "liver_interaction", ["#F9E3E7", "#F1A6B5", "#D8295F"]
    )
    xmap = {route: i for i, route in enumerate(route_order)}
    ymap = {p: i for i, p in enumerate(pair_order)}
    fig = plt.figure(figsize=(13.2, 4.8), facecolor="white")
    ax = fig.add_axes([0.08, 0.23, 0.78, 0.39])
    x = matrix["route_label"].map(xmap).astype(int)
    y = matrix["ligand_receptor_pair"].map(ymap).astype(int)
    sizes = 45 + 150 * norm_mean(values)
    scatter = ax.scatter(x, y, s=sizes, c=values, cmap=cmap_mean, norm=norm_mean,
                         edgecolor="#252525", linewidth=0.55)
    ax.set_xlim(-0.7, len(route_order) - 0.3)
    ax.set_ylim(len(pair_order) - 0.45, -0.55)
    ax.set_xticks(range(len(route_order)))
    ax.set_xticklabels(route_order, rotation=90, ha="left", va="bottom", fontsize=6.9)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=5, length=4, width=0.7)
    ax.set_yticks(range(len(pair_order)))
    ax.set_yticklabels([p.replace(" → ", " : ") for p in pair_order], fontsize=9)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", pad=8, length=4, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.grid(False)

    cax = fig.add_axes([0.31, 0.105, 0.28, 0.022])
    cb = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.set_label("CellPhoneDB mean interaction score", fontsize=8, labelpad=3)
    size_ax = fig.add_axes([0.67, 0.065, 0.23, 0.095])
    size_ax.axis("off")
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    size_ax.text(0.0, 0.92, "Interaction score", fontsize=8)
    examples = np.linspace(vmin, vmax, 3)
    for i, value in enumerate(examples):
        xpos = 0.18 + i * 0.30
        size_ax.scatter(xpos, 0.43, s=45 + 150 * norm_mean(value),
                        facecolor=cmap_mean(norm_mean(value)), edgecolor="#252525", linewidth=0.5)
        size_ax.text(xpos, 0.02, f"{value:.2f}", ha="center", fontsize=7)
    fig.text(0.035, 0.94, "I", fontsize=18, fontweight="bold")
    fig.text(0.072, 0.94, "Significant ligand–receptor interaction landscape",
             fontsize=14, fontweight="bold", color="#17253A")
    fig.text(0.072, 0.895,
             f"CellPhoneDB P < {P_THRESHOLD:g} and BH-FDR < {FDR_THRESHOLD:g}; "
             "all retained pairs; columns ordered by mean interaction score",
             fontsize=8.5, color="#667383")
    _save_publication_figure(fig, ALL_PAIR_DOT_STEM)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    routes, edges, pair_summary, ligand_summary = build_results()
    make_figures(routes, edges, pair_summary)
    draw_fig4h_alluvial(routes)
    draw_all_pair_fig4i_dotplot(routes)

    print("\nEight-ligand CellPhoneDB summary")
    print(ligand_summary.to_string(index=False))
    print(f"\nSignificant tissue routes: {len(routes)}")
    print(f"Directed tissue edges: {len(edges)}")
    print(f"Significant ligand–receptor pairs: {routes['ligand_receptor_pair'].nunique()}")
    print("\nDistinct LR-pair counts per directed tissue edge")
    print(edges.to_string(index=False))
    print(f"\nSaved main figure: {FIGURE_STEM}.png/.pdf/.svg")
    return routes, edges, pair_summary, ligand_summary


if __name__ == "__main__":
    main()
