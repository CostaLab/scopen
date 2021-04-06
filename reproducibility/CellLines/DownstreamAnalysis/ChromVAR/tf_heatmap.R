library(ComplexHeatmap)
library(chromVAR)
library(dplyr)
library(RColorBrewer)
library(Matrix)
library(matrixStats)

col_list <- list(CellType = c("H1-ESC" = "#a6cee3", "BJ" = "#1f78b4", 
                              "GM12878" = "#b2df8a", "K562" = "#33a02c",
                              "HL60" = "#fb9a99", "TF1" = "#e31a1c"))
cols <- c("H1-ESC" = "#a6cee3", "BJ" = "#1f78b4", 
          "GM12878" = "#b2df8a", "K562" = "#33a02c",
          "HL60" = "#fb9a99", "TF1" = "#e31a1c")


for (data in c("Raw", "scOpen")) {
    dev <- readRDS(sprintf("./%s/chromVAR.rds", data))
    df_dev <- as.data.frame(deviations(dev))
    df_dev[is.na(df_dev)] <- 0
    
    print(dim(df_dev))
    df_dev$sd <- rowSds(as.matrix(df_dev))
    df_dev <- df_dev[df_dev$sd > 0, ]
    df_dev$sd <- NULL
    
    print(dim(df_dev))
    
    df_dev_norm <- t(scale(t(df_dev)))
    colnames(df_dev_norm) <- colnames(df_dev)
    rownames(df_dev_norm) <- rownames(df_dev)
            
    df_anno <- read.table("../../Statistics/stat.txt", header = TRUE) %>%
        subset(., Runs %in% colnames(df_dev))
    
    rownames(df_anno) <- df_anno$Runs
    df_anno <- df_anno[order(df_anno$CellType), ]
    
    df_dev <- df_dev[, df_anno$CellType]
    
    ha = HeatmapAnnotation(
        CellType = df_anno$CellType,
        col = list(CellType = cols),
        show_annotation_name = FALSE
    )
    
    col <- colorRampPalette(rev(brewer.pal(n = 9, name = "RdYlBu")))(100)
    
    p <- Heatmap(matrix = as.matrix(df_dev_norm),
                 name = "TF",
                 cluster_rows = TRUE,
                 clustering_distance_rows = "pearson",
                 clustering_method_rows = "ward.D2",
                 cluster_columns = FALSE,
                 top_annotation = ha,
                 show_row_names = FALSE,
                 show_column_names = FALSE)
    
    tiff(file = sprintf("./%s/heatmap_tf.tiff", data), 
         width = 8, height = 8,
         units = "in", type = "cairo", res = 300)
    draw(p)
    dev.off()
    
}