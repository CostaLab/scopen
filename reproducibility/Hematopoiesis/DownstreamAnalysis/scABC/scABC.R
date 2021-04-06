library(scABC)
library(RColorBrewer)
library(devtools)
library(mclust)
library(dplyr)

num_clusters <- 8
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

}
