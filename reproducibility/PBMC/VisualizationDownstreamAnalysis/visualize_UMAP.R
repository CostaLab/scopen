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
library(chromVAR)


set.seed(42)


get_dist <- function(method, input){
    if(method == "ChromVAR"){
        dev <- readRDS(sprintf("../DownstreamAnalysis/ChromVAR/%s/chromVAR.rds", 
                               input))
        sample_cor <- getSampleCorrelation(dev,
                                           threshold = -Inf)
        sample_cor[is.na(sample_cor)] <- 0
        dist <- 1 - sample_cor
        
    } else{
        gene_activity <- read.table(sprintf("../DownstreamAnalysis/Cicero/%s/GA.txt", input),
                                    check.names = FALSE, header = TRUE)
        
        dist <- 1 - cor(gene_activity, 
                        method = "pearson")
    }
    
    dist
}

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


for (method in c("ChromVAR", "Cicero")) {
    for (input in c("Raw", "scOpen")) {
        dist <- get_dist(method = method, 
                         input = input)
        
        df_umap_out <- umap(dist,
                            min_dist = 0.3) %>%
            as.data.frame()
        
        colnames(df_umap_out) <- c("UMAP1", "UMAP2")
        df_umap_out$Cell <- colnames(dist)
        
        anno_file <- "../Statistics/stat.txt"
        df_anno <- read.table(anno_file, header = TRUE)
        df_anno <- subset(df_anno, Cell %in% colnames(dist))
        
        df_plot <- merge.data.frame(df_umap_out, df_anno, by = "Cell")
        
        p1 <- ggplot(data = df_plot, aes(x = UMAP1, y = UMAP2)) +
            geom_point(aes(color = CellType)) +
            scale_color_manual(values = cols) +
            theme_cowplot() +
            theme(legend.title = element_blank())
        
        pdf(file = sprintf("./UMAP/%s_%s.pdf", method, input),
            width = 6, height = 6)
        print(p1)
        dev.off()
        
        write.table(df_plot, file = sprintf("./UMAP/%s_%s.txt", method, input),
                    row.names = FALSE, sep = "\t", quote = FALSE)
        
        
    }
}
