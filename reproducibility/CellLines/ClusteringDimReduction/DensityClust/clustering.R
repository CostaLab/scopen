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

cols <- c("H1-ESC" = "#a6cee3", "BJ" = "#1f78b4", 
          "GM12878" = "#b2df8a", "K562" = "#33a02c",
          "HL60" = "#fb9a99", "TF1" = "#e31a1c")

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
    }else if(data == "SnapATAC"){
        obj <- readRDS("../../DimReduction/SnapATAC/x.sp.Rds")
        df <- as.data.frame(obj@smat@dmat)
        rownames(df) <- obj@barcode
        df <- t(df)
    }
    
    df_anno <- read.table("../../Statistics/stat.txt", header = TRUE) %>%
        subset(., Runs %in% colnames(df)) %>%
        rename(., Cell = Runs)
    
    df.dist <- 1 - cor(df)
    DR <- Rtsne(df.dist, is_distance = TRUE)
    df_tsne <- as.data.frame(DR$Y)
    colnames(df_tsne) <- c("tSNE1", "tSNE2")
    df_tsne$Cell <- rownames(t(df))
    
    p <- merge.data.frame(df_tsne, df_anno) %>%
        ggplot(aes(x = tSNE1, y = tSNE2, color = CellType)) +
        geom_point() +
        scale_color_manual(values = cols) +
        theme_cowplot() +
        theme(legend.title = element_blank())
    
    pdf(file = sprintf("%s/tsne.pdf", data), height = 8, width = 8)
    print(p)
    dev.off()
    
    #DRdist <- dist(DR$Y)
    DRdist <- as.dist(df.dist)
    
    library(densityClust)
    dclust <- densityClust(DRdist, gaussian=T)
    dclust <- findClusters(dclust, rho = 10, delta = 0.5)
    
    pdf(file = sprintf("%s/decision_graph.pdf", data), height = 8, width = 8)
    plot(dclust) # Inspect clustering attributes to define thresholds
    
    # Check thresholds
    options(repr.plot.width=6, repr.plot.height=6)
    plot(dclust$rho,dclust$delta,pch=20,cex=0.6,xlab='rho', ylab='delta')
    points(dclust$rho[dclust$peaks],dclust$delta[dclust$peaks],col="red",pch=20,cex=0.8)
    text(dclust$rho[dclust$peaks]-2,dclust$delta[dclust$peaks]+1.5,
         labels=dclust$clusters[dclust$peaks])
    abline(h=0.5)
    
    plotMDS(dclust)
    
    dev.off()
    
    # Add cluster information
    densityClust <- dclust$clusters
    densityClust <- as.data.frame(densityClust)
    rownames(densityClust) <- colnames(df)
    colnames(densityClust) <- 'Cluster'
    densityClust[,1] <- as.factor(densityClust[,1])
    
    densityClust$Cell <- rownames(densityClust)
    df_cluster <- merge.data.frame(densityClust, df_anno)
    
    df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType,
                                              df_cluster$Cluster))
    colnames(df_ARI) <- "ARI"
    
    write.table(df_cluster, file = sprintf("./%s/clustering.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
    write.table(df_ARI, file = sprintf("./%s/ARI.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
}