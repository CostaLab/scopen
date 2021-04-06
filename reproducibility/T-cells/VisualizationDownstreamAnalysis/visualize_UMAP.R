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

cols <- c("Jurkat_T_cell" = "#33a02c",
          "Memory_T_cell" = "#fb9a99", 
          "Naive_T_cell" = "#e31a1c",
          "Th17_T_cell" = "#cab2d6")


for (method in c("ChromVAR", "Cicero")) {
    for (input in c("Raw", "scOpen")) {
        dist <- get_dist(method = method, 
                         input = input)
        
        df_umap_out <- umap(dist,
                            min_dist = 0.3) %>%
            as.data.frame()
        
        colnames(df_umap_out) <- c("UMAP1", "UMAP2")
        df_umap_out$Runs <- colnames(dist)
        
        anno_file <- "../Statistics/stat.txt"
        df_anno <- read.table(anno_file, header = TRUE)
        df_anno <- subset(df_anno, Runs %in% colnames(dist))
        
        df_plot <- merge.data.frame(df_umap_out, df_anno, by = "Runs")
        
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
