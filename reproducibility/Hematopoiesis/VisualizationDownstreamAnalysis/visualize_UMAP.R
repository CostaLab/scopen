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

cols <- c("CLP" = "#98D9E9", "CMP" = "#FFC179", 
          "GMP" = "#FFA300", "HSC" = "#00441B",
          "LMPP" = "#00AF99", "MEP" = "#F6313E",
          "MPP" = "#46A040", "pDC" = "#C390D4")

get_df <- function(method, input){
    if(method == "ChromVAR"){
        dev <- readRDS(sprintf("../DownstreamAnalysis/ChromVAR/%s/chromVAR.rds", 
                               input))
        df <- deviations(dev)
        df[is.na(df)] <- 0
        
    } else{
        df <- read.table(sprintf("../DownstreamAnalysis/Cicero/%s/GA.txt", input),
                         check.names = FALSE, header = TRUE)
        
    }
    
    df
}


for (method in c("ChromVAR", "Cicero")) {
    for (input in c("Raw", "scOpen")) {
        df <- get_df(method = method, 
                     input = input)
        
        df_umap_out <- umap(t(df),
                            metric = "correlation",
                            min_dist = 0.3) %>%
            as.data.frame()
        
        colnames(df_umap_out) <- c("UMAP1", "UMAP2")
        df_umap_out$Runs <- colnames(df)
        
        anno_file <- "../Statistics/stat.txt"
        df_anno <- read.table(anno_file, header = TRUE)
        df_anno <- subset(df_anno, Runs %in% colnames(df))
        
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
