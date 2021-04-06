suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(BuenColors))
suppressPackageStartupMessages(library(irlba))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(Rtsne))
library(viridisLite)
library(optparse)
library(uwot)

option_list <- list( 
    make_option(c("--input_filename"),
                help="input filename"),
    make_option(c("--output_dir"), 
                help="output filename"),
    make_option(c("--output_filename"), 
                help="output filename")
)

opt <- parse_args(OptionParser(option_list=option_list))

stopifnot(file.exists(opt$input_filename))
set.seed(42)

cols <- c("CLP" = "#98D9E9", "CMP" = "#FFC179", 
          "GMP" = "#FFA300", "HSC" = "#00441B",
          "LMPP" = "#00AF99", "MEP" = "#F6313E",
          "MPP" = "#46A040", "pDC" = "#C390D4")

df <- read.table(opt$input_filename, header = TRUE)

df_umap_out <- umap(t(df), metric = "correlation",
                   min_dist = 0.3,
                   nn_method = "annoy") %>%
   as.data.frame()

colnames(df_umap_out) <- c("UMAP1", "UMAP2")
df_umap_out$Runs <- colnames(df)

anno_file <- "../../Statistics/stat.txt"
df_anno <- read.table(anno_file, header = TRUE)
df_anno <- subset(df_anno, Runs %in% colnames(df))

df_plot <- merge.data.frame(df_umap_out, df_anno, by = "Runs")

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

p2 <- ggplot(data = df_plot, aes(x = UMAP1, y = UMAP2)) +
    geom_point(aes(color = log10(NumofReads))) +
    theme_cowplot() +
    theme(legend.title = element_blank()) +
    ggtitle(opt$output_filename)

pdf(file = sprintf("%s/%s_qc.pdf", opt$output_dir, opt$output_filename),
    width = 6, height = 6)
print(p2)
dev.off()


write.table(df_plot, file = sprintf("%s/%s.txt", opt$output_dir, opt$output_filename),
            row.names = FALSE, sep = "\t", quote = FALSE)
