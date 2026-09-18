# Plot tissue-to-target-gene networks for the intervention analysis.
# Run this script from the repository root; configure paths in figure/paths.R.
# Required external inputs are listed in figure/INPUTS.md.

source("figure/paths.R")


library(ggplot2)
library(dplyr)



genes <- c("Ctsl", "Hspa8", "Rps10", "Rps28", "S100a6") # Y: 5, 4, 3, 2, 1

tissues <- c("Liver", "Muscle", "Kidney", "Skin", "BM")  # Y: 5, 4, 3, 2, 1


nodes <- data.frame(
  id = c(tissues, genes),
  label = c(tissues, genes),
  type = factor(c(rep("Tissue", 5), rep("Gene", 5)), levels = c("Tissue", "Gene")),
  x = c(rep(1, 5), rep(3, 5)),
  y = c(5, 4, 3, 2, 1,
        5, 4, 3, 2, 1)
)


edges <- data.frame(
  from = c("Liver", "Muscle", "Kidney", "BM", "BM", "BM", "BM", "Skin", "Skin", "Skin"),
  to = c("Ctsl", "Ctsl", "Rps10", "Rps10", "Hspa8", "Rps28", "S100a6", "Hspa8", "Rps28", "S100a6")
) %>% mutate(edge_id = row_number())


edges <- edges %>%
  left_join(nodes %>% select(id, x, y), by = c("from" = "id")) %>% rename(x_start = x, y_start = y) %>%
  left_join(nodes %>% select(id, x, y), by = c("to" = "id")) %>% rename(x_end = x, y_end = y)


generate_sigmoid <- function(x0, x1, y0, y1, id, n = 100) {
  x <- seq(x0, x1, length.out = n)
  x_norm <- seq(-5, 5, length.out = n)
  y_norm <- 1 / (1 + exp(-x_norm))
  y <- y0 + (y1 - y0) * y_norm
  data.frame(x = x, y = y, edge_id = id)
}

paths_list <- lapply(1:nrow(edges), function(i) {
  generate_sigmoid(edges$x_start[i], edges$x_end[i], edges$y_start[i], edges$y_end[i], edges$edge_id[i])
})
paths <- do.call(rbind, paths_list)


p <- ggplot() +


  geom_path(data = paths, aes(x = x, y = y, group = edge_id),
            color = "#C0C0C0", linewidth = 0.8, alpha = 0.6) +


  geom_tile(data = filter(nodes, type == "Tissue"), aes(x = x, y = y),
            width = 0.6, height = 0.55, fill = "#F0F0F0", color = "black", linewidth = 0.6) +


  geom_point(data = filter(nodes, type == "Gene"), aes(x = x, y = y),
             shape = 21, fill = "#4DBBD5", color = "black", size = 16, stroke = 0.6) +



  geom_text(data = filter(nodes, type == "Tissue"), aes(x = x, y = y, label = label),
            size = 3.5, color = "black", fontface = "bold", family = "sans") +

  geom_text(data = filter(nodes, type == "Gene"), aes(x = x, y = y, label = label),
            size = 3.5, color = "white", fontface = "italic", family = "sans") +


  annotate("text", x = c(1, 3), y = 5.8,
           label = c("Tested Tissues", "Conserved Targets"),
           size = 4, fontface = "bold", family = "sans", color = "#222222") +


  coord_cartesian(xlim = c(0.5, 3.5), ylim = c(0.5, 6.2)) +
  theme_void()


output_dir <- figure_output("2-8.3-shanda/1-feature/1-figure/0-4-result-5-CR-leipameisu/2-CR")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
output_path <- file.path(output_dir, "Figure_8_Bipartite_Network_Sorted.pdf")


ggsave(filename = output_path, plot = p,
       width = 4.0, height = 4.5, device = cairo_pdf)

cat("✅ 完美定制！严格按照截图基因排序的网络图已导出至:\n👉", output_path, "\n")
















###############################

library(ggplot2)
library(dplyr)



genes <- c("Ctsl", "Hspa8", "Rps10", "Rps28", "S100a6") # X: 1, 2, 3, 4, 5
tissues <- c("Liver", "Muscle", "Kidney", "Skin", "BM")  # X: 1, 2, 3, 4, 5


nodes <- data.frame(
  id = c(tissues, genes),
  label = c(tissues, genes),
  type = factor(c(rep("Tissue", 5), rep("Gene", 5)), levels = c("Tissue", "Gene")),
  x = c(1, 2, 3, 4, 5,
        1, 2, 3, 4, 5),
  y = c(rep(2, 5),
        rep(1, 5))
)


edges <- data.frame(
  from = c("Liver", "Muscle", "Kidney", "BM", "BM", "BM", "BM", "Skin", "Skin", "Skin"),
  to = c("Ctsl", "Ctsl", "Rps10", "Rps10", "Hspa8", "Rps28", "S100a6", "Hspa8", "Rps28", "S100a6")
) %>% mutate(edge_id = row_number())


edges <- edges %>%
  left_join(nodes %>% select(id, x, y), by = c("from" = "id")) %>% rename(x_start = x, y_start = y) %>%
  left_join(nodes %>% select(id, x, y), by = c("to" = "id")) %>% rename(x_end = x, y_end = y)



generate_vertical_sigmoid <- function(x0, x1, y0, y1, id, n = 100) {
  y <- seq(y0, y1, length.out = n)
  y_norm <- seq(-5, 5, length.out = n)
  x_norm <- 1 / (1 + exp(-y_norm))
  x <- x0 + (x1 - x0) * x_norm
  data.frame(x = x, y = y, edge_id = id)
}

paths_list <- lapply(1:nrow(edges), function(i) {
  generate_vertical_sigmoid(edges$x_start[i], edges$x_end[i], edges$y_start[i], edges$y_end[i], edges$edge_id[i])
})
paths <- do.call(rbind, paths_list)


p <- ggplot() +


  geom_path(data = paths, aes(x = x, y = y, group = edge_id),
            color = "#C0C0C0", linewidth = 0.8, alpha = 0.6) +


  geom_tile(data = filter(nodes, type == "Tissue"), aes(x = x, y = y),
            width = 0.8, height = 0.25, fill = "#F0F0F0", color = "black", linewidth = 0.6) +


  geom_point(data = filter(nodes, type == "Gene"), aes(x = x, y = y),
             shape = 21, fill = "#4DBBD5", color = "black", size = 16, stroke = 0.6) +



  geom_text(data = filter(nodes, type == "Tissue"), aes(x = x, y = y, label = label),
            size = 3.5, color = "black", fontface = "bold", family = "sans") +

  geom_text(data = filter(nodes, type == "Gene"), aes(x = x, y = y, label = label),
            size = 3.5, color = "white", fontface = "italic", family = "sans") +


  annotate("text", x = 3, y = c(2.35, 0.65),
           label = c("Tested Tissues", "Conserved Targets"),
           size = 4.5, fontface = "bold", family = "sans", color = "#222222") +


  coord_cartesian(xlim = c(0.4, 5.6), ylim = c(0.5, 2.5)) +
  theme_void()


output_dir <- figure_output("2-8.3-shanda/1-feature/1-figure/0-4-result-5-CR-leipameisu/3-CR")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
output_path <- file.path(output_dir, "Figure_8.1_Bipartite_Network_TopBottom.pdf")


ggsave(filename = output_path, plot = p,
       width = 6.0, height = 3.5, device = cairo_pdf)

cat("✅ 完美定制！上下布局的网络图已导出至:\n👉", output_path, "\n")
