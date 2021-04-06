library(cluster)
library(dplyr)
library(stringr)
library(SnapATAC)

df_anno <- read.table("../Statistics/stat.txt", header = TRUE)
rownames(df_anno) <- df_anno$Cell
df_anno$CellType <- stringr::str_replace_all(df_anno$CellType,
                                             c("naive_CD4_T_cells" = "1",
                                               "memory_CD4_T_cells" = "2",
                                               "naive_CD8_T_cells" = "3",
                                               "effector_CD8_T_cells" = "4",
                                               "MAIT_T_cells" = "5",
                                               "non-classical_monocytes" = "6",
                                               "classical_monocytes" = "7",
                                               "intermediate_monocytes" = "8",
                                               "memory_B_cells" = "9",
                                               "naive_B_cells" = "10",
                                               "CD56_\\(dim\\)_NK_cells" = "11",
                                               "CD56_\\(bright\\)_NK_cells" = "12",
                                               "myeloid_DC" = "13",
                                               "plasmacytoid_DC" = "14"))
df_anno$CellType <- as.numeric(df_anno$CellType)

get_dr_matrix <- function(method){
    if(method == "scOpen"){
        dr_mat <- read.table("../Imputation/scOpen/scOpen_barcodes.txt",
                             header = TRUE)
        dr_mat <- t(dr_mat)
        
    } else if(method == "SnapATAC"){
        obj <- readRDS("../DimReduction/SnapATAC/x.sp.Rds")
        dr_mat <- obj@smat@dmat
        rownames(dr_mat) <- obj@metaData$barcode
    } else if(method == "cisTopic"){
        dr_mat <- read.table("../Imputation/cisTopic/documents.txt",
                             header = TRUE, row.names = 1)
        dr_mat <- t(dr_mat)
    } else if(method == "Cusanovich2018"){
        obj <- readRDS("../DimReduction/Cusanovich2018/obj.Rds")
        dr_mat <- as.data.frame(obj@reductions$lsi@cell.embeddings)
    }
    return(dr_mat)
}

for (method in c("scOpen", "cisTopic", "Cusanovich2018", "SnapATAC")) {
    dr_mat <- get_dr_matrix(method = method)
    df_anno <- df_anno[rownames(dr_mat), ]
    
    dist = 1 - cor(t(dr_mat))
    
    si <- silhouette(x = df_anno$CellType, 
                     dmatrix = dist)
    
    saveRDS(si, sprintf("%s.Rds", method))
    
    pdf(sprintf("%s.pdf", method), height = 8, width = 8)
    plot(si)
    dev.off()

}
