library(optparse)
library(cisTopic)
library(methods)
library(stringr)
library(dplyr)
library(cowplot)
library(gridExtra)
library(mclust)
library(ggplot2)
library(Rtsne)
library(Seurat)


data_list <- c("cisTopic", "scOpen", "Cusanovich2018", "SnapATAC")


for (data in data_list) {
    set.seed(42)
    
    if(!dir.exists(data)){
        dir.create(data)
    }
    
    if(data == "cisTopic"){
        df <- read.table("../../Imputation/cisTopic/documents.txt", header = TRUE)
    } else if (data == "scOpen"){
        df <- read.table("../../Imputation/scOpen/scOpen_barcodes.txt", 
                         header = TRUE, row.names = 1)
    } else if(data == "Cusanovich2018"){
        obj <- readRDS("../../DimReduction/Cusanovich2018/obj.Rds")
        df <- as.data.frame(obj@reductions$lsi@cell.embeddings)
        df <- t(df)
    } else if(data == "SnapATAC"){
        obj <- readRDS("../../DimReduction/SnapATAC/x.sp.Rds")
        df <- as.data.frame(obj@smat@dmat)
        rownames(df) <- obj@barcode
        df <- t(df)
    }
    
    df_dist <- as.dist(1 - cor(df))
    
    snn <- FindNeighbors(object = df_dist)$snn
    df_cluster <- FindClusters(object = snn,
                               random.seed = 42) %>%
        as.data.frame()
    
    colnames(df_cluster) <- "Cluster"
    df_cluster$Cell <- rownames(df_cluster)
    
    df_anno <- read.table("../../Statistics/stat.txt", header = TRUE) %>%
        subset(., Runs %in% rownames(df_cluster)) %>%
        rename(., Cell = Runs)
    
    df_cluster <- merge.data.frame(df_cluster, df_anno)
    df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType,
                                              df_cluster$Cluster))
    colnames(df_ARI) <- "ARI"
    
    write.table(df_ARI, file = sprintf("./%s/ARI.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
}


