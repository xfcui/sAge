# Compare gene-set Jaccard and GO semantic similarity across mouse tissues.
# Run this script from the repository root; configure paths in figure/paths.R.
# Required external inputs are listed in figure/INPUTS.md.

source("figure/paths.R")

# ==============================================================================


# ==============================================================================

graphics.off()

if (!requireNamespace("pheatmap", quietly = TRUE)) install.packages("pheatmap")
if (!requireNamespace("proxy", quietly = TRUE)) install.packages("proxy")
if (!requireNamespace("RColorBrewer", quietly = TRUE)) install.packages("RColorBrewer")

library(pheatmap)
library(proxy)
library(RColorBrewer)
library(clusterProfiler)
library(org.Mm.eg.db)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(GOSemSim)
library(dplyr)
library(grid)



final_output_dir <- figure_output("2-8.3-shanda/1-feature/1-figure/0-1-result-2-mouse-gene/4-GO-upset-jaccard")
if (!dir.exists(final_output_dir)) dir.create(final_output_dir, recursive = TRUE)


pub_colors <- colorRampPalette(c("#FFFFFF", "#FFF7BC", "#FC8D59", "#E31A1C", "#800026"))(100)
pub_breaks <- seq(0, 1, length.out = 101)


grid_border_color <- NA

leg_breaks <- seq(0, 1, by = 0.2)
leg_labels <- c("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")


message(">>> [Init] 正在初始化 GO 语义拓扑数据库 (GOSemSim)...")
mmGO <- godata('org.Mm.eg.db', ont="BP", computeIC=FALSE)
hsGO <- godata('org.Hs.eg.db', ont="BP", computeIC=FALSE)

map_genes_to_entrez <- function(gene_list, org_db) {
  tryCatch({
    all_genes <- AnnotationDbi::select(org_db, keys = keys(org_db, keytype="ENTREZID"), columns = c("SYMBOL"), keytype = "ENTREZID")
    all_genes$SYMBOL_UPPER <- toupper(all_genes$SYMBOL)
    matched_data <- all_genes[all_genes$SYMBOL_UPPER %in% toupper(gene_list), ]
    if (nrow(matched_data) == 0) return(NULL)
    return(unique(matched_data$ENTREZID))
  }, error = function(e) { return(NULL) })
}

run_go_enrichment <- function(entrez_ids, org_db) {
  if (is.null(entrez_ids) || length(entrez_ids) == 0) return(NULL)
  ego <- enrichGO(gene = entrez_ids, OrgDb = org_db, ont = "BP", pAdjustMethod = "BH", pvalueCutoff = 0.05, qvalueCutoff = 0.2)
  if (is.null(ego) || nrow(ego) == 0) return(NULL)
  return(ego@result$ID[ego@result$p.adjust < 0.05])
}

# ==============================================================================

# ==============================================================================
input_dir_mouse <- figure_input("2-8.3-shanda/1-feature/9-Final_Segmented_Genes")

if (dir.exists(input_dir_mouse)) {
  message("\n========================================================")
  message(">>> [Mouse] 开始构建小鼠图谱...")
  setwd(input_dir_mouse)

  file_list_m <- list.files(pattern = "\\.txt$")
  gene_list_m_raw <- list()
  pathway_list_m_raw <- list()

  for (f in file_list_m) {
    tissue <- gsub("_Knee_Genes.*", "", f) %>% gsub(".txt", "", .) %>% gsub("_", " ", .)
    genes <- readLines(f) %>% unique()
    genes <- toupper(genes[genes != "" & !grepl("SOURCE", toupper(genes))])

    if (length(genes) > 0) {
      gene_list_m_raw[[tissue]] <- genes
      gene_ids <- map_genes_to_entrez(genes, org.Mm.eg.db)
      if (!is.null(gene_ids)) {
        go_ids <- run_go_enrichment(gene_ids, org.Mm.eg.db)
        if (!is.null(go_ids) && length(go_ids) > 0) pathway_list_m_raw[[tissue]] <- go_ids
      }
    }
  }

  valid_tissues_m <- sort(intersect(names(gene_list_m_raw), names(pathway_list_m_raw)))
  gene_list_m <- gene_list_m_raw[valid_tissues_m]
  pathway_list_m <- pathway_list_m_raw[valid_tissues_m]
  n_m <- length(valid_tissues_m)

  if(n_m > 2) {
    all_m_g <- unique(unlist(gene_list_m))
    mat_m_g <- sapply(gene_list_m, function(x) as.integer(all_m_g %in% x))
    sim_m_g <- as.matrix(proxy::simil(as.matrix(t(mat_m_g)), method = "Jaccard"))
    diag(sim_m_g) <- 1.0

    sim_m_p <- matrix(NA, nrow=n_m, ncol=n_m, dimnames=list(valid_tissues_m, valid_tissues_m))
    message("  -> 正在计算小鼠 GO 语义拓扑网络...")
    for(i in 1:n_m) {
      for(j in 1:n_m) {
        if(i == j) { sim_m_p[i,j] <- 1.0
        } else if (i < j) {
          score <- mgoSim(pathway_list_m[[i]], pathway_list_m[[j]], semData=mmGO, measure="Wang", combine="BMA")
          sim_m_p[i,j] <- score; sim_m_p[j,i] <- score
        }
      }
    }

    sim_m_g[is.na(sim_m_g) | is.nan(sim_m_g) | is.infinite(sim_m_g)] <- 0
    sim_m_p[is.na(sim_m_p) | is.nan(sim_m_p) | is.infinite(sim_m_p)] <- 0

    master_dist_m <- as.dist(1 - sim_m_p)
    master_tree_m <- hclust(master_dist_m, method = "ward.D2")


    p_width <- 90 / 25.4
    p_height <- 80 / 25.4

    message("  -> 正在渲染小鼠图谱...")


    out_pathway_m <- file.path(final_output_dir, "Fig1B_Mouse_Pathway_Semantic.pdf")
    pdf(out_pathway_m, width = p_width, height = p_height, family = "Helvetica", useDingbats = FALSE)
    pheatmap::pheatmap(sim_m_p, color = pub_colors, breaks = pub_breaks,
                       display_numbers = FALSE,
                       fontsize_number = 5,
                       fontsize = 6, fontsize_row = 6, fontsize_col = 6,
                       treeheight_row = 15, treeheight_col = 15,
                       cluster_rows = master_tree_m, cluster_cols = master_tree_m,
                       border_color = grid_border_color,
                       main = "", angle_col = "45",
                       legend_breaks = leg_breaks, legend_labels = leg_labels)

    grid::grid.text("Semantic Similarity",
                    x = grid::unit(1, "npc") - grid::unit(0.4, "inches"),
                    y = grid::unit(1, "npc") - grid::unit(0.15, "inches"),
                    gp = grid::gpar(fontsize = 6, fontfamily = "Helvetica"))
    dev.off()


    out_gene_m <- file.path(final_output_dir, "Fig1A_Mouse_Gene_Jaccard.pdf")
    pdf(out_gene_m, width = p_width, height = p_height, family = "Helvetica", useDingbats = FALSE)
    pheatmap::pheatmap(sim_m_g, color = pub_colors, breaks = pub_breaks,
                       display_numbers = FALSE,
                       fontsize_number = 5,
                       fontsize = 6, fontsize_row = 6, fontsize_col = 6,
                       treeheight_row = 15, treeheight_col = 15,
                       cluster_rows = master_tree_m, cluster_cols = master_tree_m,
                       border_color = grid_border_color,
                       main = "", angle_col = "45",
                       legend_breaks = leg_breaks, legend_labels = leg_labels)

    grid::grid.text("Jaccard Similarity",
                    x = grid::unit(1, "npc") - grid::unit(0.4, "inches"),
                    y = grid::unit(1, "npc") - grid::unit(0.15, "inches"),
                    gp = grid::gpar(fontsize = 6, fontfamily = "Helvetica"))
    dev.off()

    message("  -> [成功] 小鼠双对比图谱已安全保存！")
  } else {
    message("⚠️ 警告：满足要求的小鼠组织数量太少（<3），无法进行聚类分析！")
  }
}

message("\n========================================================")
message(">>> 🎉 终极任务完美收官！110x110mm 无边框、全数据展示版已保存。")
message(">>> 结果请查看: ", final_output_dir)
message("========================================================")
