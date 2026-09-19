# Source this file from the repository root before an analysis changes getwd().
figure_repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

figure_resolve_root <- function(value) {
  absolute <- grepl("^(/|[A-Za-z]:[/\\\\])", value)
  path <- if (absolute) value else file.path(figure_repo_root, value)
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

figure_input_root <- figure_resolve_root(
  Sys.getenv("SAGE_FIGURE_INPUT_ROOT", unset = "figure_inputs")
)
figure_output_root <- figure_resolve_root(
  Sys.getenv("SAGE_FIGURE_OUTPUT_ROOT", unset = "figure_outputs")
)

figure_input <- function(relative) {
  file.path(figure_input_root, relative)
}

figure_output <- function(relative) {
  path <- file.path(figure_output_root, relative)
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  path
}

figure_font <- function() {
  Sys.getenv("SAGE_FIGURE_FONT", unset = "")
}
