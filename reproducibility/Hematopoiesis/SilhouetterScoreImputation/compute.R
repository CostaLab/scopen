library(cluster)
library(dplyr)
library(stringr)
library(SnapATAC)

df_anno <- read.table("../Statistics/stat.txt", header = TRUE) %>%
    rename(., Cell = Runs)
rownames(df_anno) <- df_anno$Cell
df_anno$CellType <- stringr::str_replace_all(df_anno$CellType,
                                             c("CLP" = "1", "CMP" = "2", 
                                               "GMP" = "3", "HSC" = "4",
                                               "LMPP" = "5", "MEP" = "6",
                                               "MPP" = "7", "pDC" = "8"))
df_anno$CellType <- as.numeric(df_anno$CellType)

method_list <- c('scImpute','MAGIC', 'SAVER', 'cisTopic',
                 'DCA', 'scBFA', 'SCALE', 'Raw', 'PCA',
                 'scOpen')

for (method in method_list) {
    if (method == "Raw"){
        filename <- "../TagCount/TagCount.txt"
    } else if (method == "scImpute"){
        filename <- "../Imputation/scImpute/scimpute_count.txt"
    } else if (method == "SCALE"){
        filename <- "../Imputation/SCALE/imputed_data.txt"
    } else if (method == "DCA"){
        filename <- "../Imputation/DCA/mean.tsv"
    } else{
        filename <- sprintf("../Imputation/%s/%s.txt",
                            method, method)
    }
    mat <- read.table(filename, header = TRUE)
    df_anno <- df_anno[colnames(mat), ]
    dist = 1 - cor(mat)
    
    si <- silhouette(x = df_anno$CellType, 
                     dmatrix = dist)
    
    saveRDS(si, sprintf("%s.Rds", method))
    
    pdf(sprintf("%s.pdf", method), height = 8, width = 8)
    plot(si)
    dev.off()

}