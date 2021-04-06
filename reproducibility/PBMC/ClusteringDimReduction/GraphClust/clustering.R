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


cols <- c("Jurkat_T_cell" = "#33a02c",
          "Memory_T_cell" = "#fb9a99", 
          "Naive_T_cell" = "#e31a1c",
          "Th17_T_cell" = "#cab2d6")

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
    
    # here we used clustering approach from SnapATAC
    x.sp <- readRDS("../../DimReduction/SnapATAC/x.sp.Rds")
    
    x.sp@smat@dmat <- as.matrix(t(df))
    
    library(matrixStats)
    
    x.sp@smat@sdev <- rowSds(as.matrix(df))
    x.sp = runKNN(
        obj=x.sp,
        eigs.dims=1:nrow(df),
        k=15
    )
    x.sp=runCluster(
        obj=x.sp,
        tmp.folder=tempdir(),
        louvain.lib="R-igraph",
        seed.use=10
    )
    
    x.sp@metaData$Cluster = x.sp@cluster
    
    df_cluster <- as.data.frame(x.sp@metaData)
    df_cluster$barcode <- colnames(df)
    
    df_anno <- read.table("../../Statistics/stat.txt", header = TRUE) %>%
        subset(., Cell %in% df_cluster$barcode) %>%
        rename(., barcode = Cell)
    
    df_cluster <- merge.data.frame(df_cluster, df_anno)
    df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType,
                                              df_cluster$Cluster))
    colnames(df_ARI) <- "ARI"
    
    write.table(df_ARI, file = sprintf("./%s/ARI.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
}


