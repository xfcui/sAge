# Plot multi-tissue GO enrichment results.
# Run this script from the repository root; configure paths in figure/paths.R.
# Required external inputs are listed in figure/INPUTS.md.

source("figure/paths.R")

# ==============================================================================


# ==============================================================================

graphics.off()


library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(ggplot2)
library(forcats)
library(scales)
library(stats)


input_dir        <- figure_input("2-8.3-shanda/1-feature/11-1-Enrichment_R_Results")
final_output_dir <- figure_output("2-8.3-shanda/1-feature/1-figure/4-GO-MultiTissue")
if (!dir.exists(final_output_dir)) dir.create(final_output_dir, recursive = TRUE)
output_pdf <- file.path(final_output_dir, "1.3-Figure_Waterfall_DotPlot_NatureFinal_v9.1.pdf")

# ==============================================================================

# ==============================================================================
fix_bio_terms <- function(x) {
  x <- str_to_sentence(x)
  bio_terms <- c("mRNA", "RNA", "rRNA", "tRNA", "snRNA", "DNA", "ATP", "ADP", "NF-kB", "TGF", "EGF", "T cell", "B cell")
  for (term in bio_terms) {
    x <- str_replace_all(x, regex(paste0("\\b", term, "\\b"), ignore_case = TRUE), term)
  }
  return(x)
}

# ==============================================================================

# ==============================================================================
fix_tissue_names <- function(x) {
  x <- str_replace_all(x, "_", " ")
  x <- str_to_title(x)
  x <- str_replace_all(x, "\\bAnd\\b", "and")
  return(x)
}


message(">>> 正在整合跨组织富集数据...")
csv_files <- list.files(input_dir, pattern = "_GO_BP_Table\\.csv$", full.names = TRUE)

all_go_data <- lapply(csv_files, function(file) {
  tissue_name <- str_remove(basename(file), "_GO_BP_Table\\.csv")
  df <- read_csv(file, show_col_types = FALSE)
  if (nrow(df) > 0) {
    df$Tissue        <- fix_tissue_names(tissue_name)
    df$LogP          <- -log10(df$p.adjust)
    df$GeneRatioNum  <- sapply(str_split(df$GeneRatio, "/"), function(x) as.numeric(x[1]) / as.numeric(x[2]))
    return(df)
  }
  return(NULL)
}) %>% bind_rows()


shared_pathways <- all_go_data %>%
  filter(p.adjust < 0.01) %>%
  group_by(Description) %>%
  summarise(Tissue_Count = n_distinct(Tissue)) %>%
  filter(Tissue_Count >= 4) %>%
  pull(Description)

top_specific_pathways <- all_go_data %>%
  filter(!Description %in% shared_pathways) %>%
  group_by(Tissue) %>%
  slice_max(order_by = LogP, n = 2, with_ties = FALSE) %>%
  pull(Description)

target_pathways <- unique(c(shared_pathways, top_specific_pathways))
plot_df <- all_go_data %>% filter(Description %in% target_pathways)

plot_df$Description_Clean <- fix_bio_terms(plot_df$Description) %>%
  str_wrap(width = 40)


message(">>> 正在执行瀑布流截断与组织聚类算法...")

pathway_rank <- plot_df %>%
  group_by(Description_Clean) %>%
  summarise(Prevalence = n(), MaxIntensity = max(LogP)) %>%
  arrange(desc(Prevalence), desc(MaxIntensity))

top_20_pathways <- head(pathway_rank$Description_Clean, 20)
plot_df <- plot_df %>% filter(Description_Clean %in% top_20_pathways)
plot_df$Description_Clean <- factor(plot_df$Description_Clean, levels = rev(top_20_pathways))


wide_mat <- plot_df %>%
  dplyr::select(Tissue, Description_Clean, LogP) %>%
  pivot_wider(names_from = Tissue, values_from = LogP, values_fill = list(LogP = 0)) %>%
  as.data.frame()
rownames(wide_mat)      <- wide_mat$Description_Clean
wide_mat$Description_Clean <- NULL

hc                <- hclust(dist(t(wide_mat)), method = "ward.D2")
clustered_tissues <- colnames(wide_mat)[hc$order]
plot_df$Tissue    <- factor(plot_df$Tissue, levels = clustered_tissues)


cap_value           <- 8
plot_df$LogP_Capped <- pmin(plot_df$LogP, cap_value)

# ==============================================================================

# ==============================================================================
pub_color_scale <- c("#FEEDDE", "#FDAE6B", "#E31A1C", "#800026")

max_gr <- max(plot_df$GeneRatioNum, na.rm = TRUE)
gr_breaks <- c(0.05, 0.10, 0.15, 0.20, 0.25)
gr_breaks <- gr_breaks[gr_breaks <= max_gr + 0.03]

p_width  <- 120 / 25.4
p_height <- 110 / 25.4

p <- ggplot(plot_df, aes(x = Tissue, y = Description_Clean)) +

  geom_vline(xintercept = seq_along(levels(plot_df$Tissue)), color = "grey93", linewidth = 0.25) +

  geom_point(aes(size = GeneRatioNum, fill = LogP_Capped),
             shape = 21, color = "grey35", stroke = 0.35, alpha = 0.95) +

  scale_x_discrete(expand = expansion(add = c(0.6, 0.6))) +
  scale_y_discrete(expand = expansion(add = c(0.6, 0.6))) +

  scale_fill_gradientn(
    colors = pub_color_scale,
    name   = bquote(-log[10] ~ italic(P)[adj]),
    limits = c(0, cap_value),
    oob    = scales::squish,
    guide  = guide_colorbar(
      barwidth  = unit(0.35, "cm"),
      barheight = unit(2.2, "cm"),
      ticks.linewidth = 0.4,
      frame.colour = NA
    )
  ) +

  scale_size_continuous(
    name   = "Gene Ratio",
    range  = c(1, 3.8),
    breaks = gr_breaks
  ) +

  guides(
    size = guide_legend(
      override.aes = list(fill = "grey60", color = "grey35", stroke = 0.35)
    )
  ) +

  theme_bw(base_size = 6.5, base_family = "Helvetica") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color = "black", size = 6),
    axis.text.y = element_text(color = "black", size = 6, lineheight = 0.82),
    axis.title  = element_blank(),

    panel.grid.major.y = element_line(color = "grey88", linewidth = 0.3, linetype = "dashed"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(color = "black", linewidth = 0.5, fill = NA),

    legend.position    = "right",
    legend.key.size    = unit(0.3, "cm"),
    legend.title       = element_text(face = "bold", size = 6),
    legend.text        = element_text(size = 5.5),
    legend.box.margin  = margin(l = 2),
    legend.spacing.y   = unit(0.15, "cm"),

    plot.margin = margin(t = 2, r = 2, b = 2, l = 2, unit = "mm")
  )


cairo_pdf(output_pdf, width = p_width, height = p_height, family = "Helvetica", onefile = FALSE)
print(p)
dev.off()

message("--- ✅ v9.1 纯净版已生成！（色条黑框已移除）---")
message("--- 📁 路径: ", output_pdf, " ---")
