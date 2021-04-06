library(cluster)
library(dplyr)
library(stringr)
library(SnapATAC)
library(chromVAR)

df_anno <- read.table("../Statistics/stat.txt", header = TRUE) %>%
    rename(., Cell = Runs)
rownames(df_anno) <- df_anno$Cell
df_anno$CellType <- stringr::str_replace_all(df_anno$CellType,
                                             c("CLP" = "1", "CMP" = "2", 
                                               "GMP" = "3", "HSC" = "4",
                                               "LMPP" = "5", "MEP" = "6",
                                               "MPP" = "7", "pDC" = "8"))
df_anno$CellType <- as.numeric(df_anno$CellType)

for (method in c("ChromVAR", "Cicero")) {
    if(method == "ChromVAR"){
        for(data in c("Raw", "scOpen")){
            dev <- readRDS(sprintf("../DownstreamAnalysis/ChromVAR/%s/chromVAR.rds", data))
            sample_cor <- getSampleCorrelation(dev,
                                               threshold = -Inf)
            sample_cor[is.na(sample_cor)] <- 0
            
            df_anno <- df_anno[colnames(sample_cor), ]
            
            si <- silhouette(x = df_anno$CellType, 
                             dmatrix = 1 - sample_cor)
            
            saveRDS(si, sprintf("%s_%s.Rds", method, data))
            
            pdf(sprintf("%s_%s.pdf", method, data), height = 8, width = 8)
            plot(si)
            dev.off()
            
        }
    }
    else if (method == "Cicero"){
        for(data in c("Raw", "scOpen")){
            df <- read.table(sprintf("../DownstreamAnalysis/Cicero/%s/GA.txt", data),
                             header = TRUE)
            
            df_anno <- df_anno[colnames(df), ]
            
            dist <- 1 - cor(df)
            
            si <- silhouette(x = df_anno$CellType, 
                             dmatrix = dist)
            
            saveRDS(si, sprintf("%s_%s.Rds", method, data))
            
            pdf(sprintf("%s_%s.pdf", method, data), height = 8, width = 8)
            plot(si)
            dev.off()
            
        }
    }
}
