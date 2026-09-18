# Build mouse and human tissue selected-gene UpSet plots.
# Run this script from the repository root; configure paths in figure/paths.R.
# Required external inputs are listed in figure/INPUTS.md.

source("figure/paths.R")



if (!require("ComplexHeatmap")) BiocManager::install("ComplexHeatmap")
library(ComplexHeatmap)
library(tidyverse)
library(grid)


folder_path <- figure_input("2-8.3-shanda/1-feature/9-Final_Segmented_Genes/")
file_list <- list.files(path = folder_path, pattern = "\\.txt$", full.names = TRUE)

gene_list <- lapply(file_list, readLines)


clean_names <- gsub("_Knee.*", "", basename(file_list))
clean_names <- gsub("_", " ", clean_names)


names(gene_list) <- clean_names


m = make_comb_mat(gene_list)
m = m[comb_size(m) >= 3]


col_main = "#333333"
col_set  = "#0072B2"


pdf(figure_output("2-8.3-shanda/1-feature/1-figure/1-mouse-gene-upset/1-Mouse_Tissue_Aging_UpSet.pdf"),
    width = 4.53, height = 2.95)

ht = UpSet(m,

           top_annotation = upset_top_annotation(m,
                                                 add_numbers = TRUE,
                                                 numbers_rot = 0,
                                                 numbers_gp = gpar(fontsize = 5, fontfamily = "Helvetica"),
                                                 annotation_name_gp = gpar(fontsize = 6, fontfamily = "Helvetica"),
                                                 annotation_name_rot = 90,

                                                 axis_param = list(gp = gpar(fontsize = 5, fontfamily = "Helvetica")),
                                                 gp = gpar(fill = col_main, col = col_main),
                                                 height = unit(1.5, "cm")),


           right_annotation = upset_right_annotation(m,
                                                     add_numbers = TRUE,
                                                     numbers_rot = 0,
                                                     numbers_gp = gpar(fontsize = 5, fontfamily = "Helvetica"),
                                                     annotation_name_gp = gpar(fontsize = 6, fontfamily = "Helvetica"),
                                                     axis_param = list(labels_rot = 0,
                                                                       gp = gpar(fontsize = 5, fontfamily = "Helvetica")),
                                                     gp = gpar(fill = col_set, col = "white"),
                                                     width = unit(1.8, "cm")),


           comb_col = col_main,
           pt_size = unit(1.5, "mm"),
           lwd = 0.75,


           comb_order = order(comb_degree(m), -comb_size(m)),
           set_order = order(set_size(m), decreasing = TRUE),


           row_names_side = "left",
           row_names_gp = gpar(fontsize = 6, fontfamily = "Helvetica"),
           bg_col = c("#F2F2F2", "#E6E6E6")
)


draw(ht)


dev.off()












if (!require("ComplexHeatmap")) BiocManager::install("ComplexHeatmap")
library(ComplexHeatmap)
library(tidyverse)
library(RColorBrewer)
library(tools)
library(grid)

# ==========================================

# ==========================================
folder_path <- figure_input("2-8.3-shanda/1-feature/1-human-guaidian-choose-gene/Gene_Lists/")
file_list <- list.files(path = folder_path, pattern = "\\.txt$", full.names = TRUE)

gene_list <- lapply(file_list, readLines)


raw_names <- basename(file_list)


clean_names <- gsub("_Knee.*", "", raw_names)


clean_names <- gsub("^type_[0-9]+_", "", clean_names)


clean_names <- gsub("_", " ", clean_names)
clean_names <- tools::toTitleCase(clean_names)

names(gene_list) <- clean_names

# ==========================================

# ==========================================
m = make_comb_mat(gene_list)


m = m[comb_size(m) >= 3]

# ==========================================

# ==========================================

col_main = "#333333"
col_set  = "#0072B2"

# ==========================================

# ==========================================
pdf(figure_output("2-8.3-shanda/1-feature/1-figure/0-1-result-2-mouse-gene/1-mouse-gene-upset/1-Human_Tissue_Aging_UpSet.pdf"),
    width = 4.53, height = 2.95)

ht = UpSet(m,

           top_annotation = upset_top_annotation(m,
                                                 add_numbers = TRUE,
                                                 numbers_rot = 0,

                                                 numbers_gp = gpar(fontsize = 5, fontfamily = "Helvetica"),
                                                 annotation_name_gp = gpar(fontsize = 6, fontfamily = "Helvetica"),
                                                 annotation_name_rot = 90,
                                                 axis_param = list(gp = gpar(fontsize = 5, fontfamily = "Helvetica")),
                                                 gp = gpar(fill = col_main, col = col_main),
                                                 height = unit(1.5, "cm")),


           right_annotation = upset_right_annotation(m,
                                                     add_numbers = TRUE,
                                                     numbers_rot = 0,
                                                     numbers_gp = gpar(fontsize = 5, fontfamily = "Helvetica"),
                                                     annotation_name_gp = gpar(fontsize = 6, fontfamily = "Helvetica"),
                                                     axis_param = list(labels_rot = 0,
                                                                       gp = gpar(fontsize = 5, fontfamily = "Helvetica")),
                                                     gp = gpar(fill = col_set, col = "white"),
                                                     width = unit(1.8, "cm")),


           comb_col = col_main,
           pt_size = unit(1.5, "mm"),
           lwd = 0.75,


           comb_order = order(comb_degree(m), -comb_size(m)),
           set_order = order(set_size(m), decreasing = TRUE),


           row_names_side = "left",
           row_names_gp = gpar(fontsize = 6, fontfamily = "Helvetica"),
           bg_col = c("#F2F2F2", "#E6E6E6")
)


draw(ht)


dev.off()

print("Human UpSet plot generated successfully at Top-Tier standards!")
