library(fclust)
library(cluster)
library(e1071)
library(foreach)
library(optparse)
library(Rtsne)
library(patchwork)
library(ggplot2)
library(cowplot)
library(gridExtra)
library(uwot)
library(dplyr)


option_list <- list( 
    make_option(c("--input"),
                help="input filename"),
    make_option(c("--output_dir"), 
                help="output filename"),
    make_option(c("--output_filename"), 
                help="output filename")
)

opt <- parse_args(OptionParser(option_list=option_list))

cols <- c("naive_CD4_T_cells" = "#fb9a99",
          "memory_CD4_T_cells" = "#e31a1c",
          "naive_CD8_T_cells" = "#fdbf6f",
          "effector_CD8_T_cells" = "#ff7f00",
          "MAIT_T_cells" = "#fb8072",
          "non-classical_monocytes" = "#6a3d9a",
          "classical_monocytes" = "#cab2d6",
          "intermediate_monocytes" = "#bc80bd",
          "memory_B_cells" = "#8dd3c7",
          "naive_B_cells" = "#b3de69",
          "CD56_(dim)_NK_cells" = "#a6cee3",
          "CD56_(bright)_NK_cells" = "#1f78b4",
          "myeloid_DC" = "#b15928",
          "plasmacytoid_DC" = "#33a02c")


stopifnot(file.exists(opt$input))
df <- read.table(opt$input, header = TRUE)
df_umap_out <- umap(t(df), metric = "correlation",
                    min_dist = 0.3,
                    nn_method = "annoy") %>%
    as.data.frame()

colnames(df_umap_out) <- c("UMAP1", "UMAP2")
df_umap_out$Cell <- colnames(df)

anno_file <- "../../Statistics/stat.txt"
df_anno <- read.table(anno_file, header = TRUE)
df_anno <- subset(df_anno, Cell %in% colnames(df))

df_plot <- merge.data.frame(df_umap_out, df_anno, by = "Cell")

p1 <- ggplot(data = df_plot, aes(x = UMAP1, y = UMAP2)) +
    geom_point(aes(color = CellType)) +
    scale_color_manual(values = cols) +
    theme_cowplot() +
    theme(legend.title = element_blank()) +
    ggtitle(opt$output_filename)

pdf(file = sprintf("%s/%s.pdf", opt$output_dir, opt$output_filename),
    width = 6, height = 6)
print(p1)
dev.off()

write.table(df_plot, file = sprintf("%s/%s.txt", opt$output_dir, opt$output_filename),
            row.names = FALSE, sep = "\t", quote = FALSE)
