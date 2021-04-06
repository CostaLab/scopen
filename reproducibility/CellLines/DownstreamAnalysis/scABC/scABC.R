library(scABC)
library(RColorBrewer)
library(devtools)
library(mclust)
library(dplyr)

num_clusters <- 6
data_list <- c("Raw", "scOpen")

for (data in data_list) {
    if(data == "Raw"){
        input_file <- "../TagCount/TagCount.txt"
    } else{
        input_file <- "../Imputation/scOpen/scOpen.txt"
    }
    
    if(!dir.exists(data)){
        dir.create(data)
    }
    
    df <- read.table(input_file, header = TRUE)
    df_anno <- read.table("../Statistics/stat.txt", header = TRUE) %>%
        subset(., Runs %in% colnames(df)) %>%
        rename(., Cell = Runs)
    
    rownames(df_anno) <- df_anno$Cell
    df_anno <- subset(df_anno, select = c("CellType"))
    
    weights <- as.numeric(unlist(apply(df, 2, mean)))
    
    LandMarks <- computeLandmarks(ForeGround = as.matrix(df), 
                                 weights = weights, 
                                 nCluster = num_clusters, 
                                 nTop = 5000)
    LandMarkAssignments <- assign2landmarks(df, LandMarks)
    df_cluster <- as.data.frame.integer(LandMarkAssignments)
    colnames(df_cluster) <- c("Cluster")
    
    df_cluster$Cell <- rownames(df_cluster)
    df_anno$Cell <- rownames(df_anno)
    df_cluster <- merge.data.frame(df_cluster,
                                   df_anno,
                                   by = "Cell")
    
    df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType, 
                                              df_cluster$Cluster))
    colnames(df_ARI) <- "ARI"
    
    write.table(df_cluster, file = sprintf("./%s/clustering.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
    write.table(df_ARI, file = sprintf("./%s/ARI.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
    LandmarkCorrelation = cbind(apply(df, 2, function(x) cor(x, LandMarks[,1], method = 'spearman')),
                                apply(df, 2, function(x) cor(x, LandMarks[,2], method = 'spearman')),
                                apply(df, 2, function(x) cor(x, LandMarks[,3], method = 'spearman')),
                                apply(df, 2, function(x) cor(x, LandMarks[,4], method = 'spearman')),
                                apply(df, 2, function(x) cor(x, LandMarks[,5], method = 'spearman')),
                                apply(df, 2, function(x) cor(x, LandMarks[,6], method = 'spearman')))
    
    write.table(LandmarkCorrelation, 
                file = sprintf("./%s/correlation.txt", data), 
                sep = "\t", 
                quote = FALSE, col.names = FALSE)
    
    LandmarkCorrelation <- as.data.frame(t(apply(LandmarkCorrelation, 1, 
                                                 FUN = function(X) (X - min(X))/diff(range(X))))) 

    colnames(LandmarkCorrelation) <- c("1", "2", "3", "4", "5", "6")
    
    df_anno$CellType <- factor(df_anno$CellType, 
                               levels = c("BJ", "GM12878", "H1-ESC", "HL60", "K562", "TF1"))
    
    cols <- colorRampPalette(c("white", "red"), space = "rgb")(256)
    col_list1 <- list(CellType = c("H1-ESC" = "#a6cee3", "BJ" = "#1f78b4", 
                                   "GM12878" = "#b2df8a", "K562" = "#33a02c",
                                   "HL60" = "#fb9a99", "TF1" = "#e31a1c"))
    
    col_list2 <- list(Cluster = c("1" = "#7fc97f", "2" = "#beaed4", 
                                  "3" = "#fdc086", "4" = "#ffff99",
                                  "5" = "#bf5b17", "6" = "#f0027f"))
    

}
