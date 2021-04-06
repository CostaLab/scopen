library(cluster)
library(dplyr)
library(stringr)
library(SnapATAC)
library(optparse)

option_list <- list( 
    make_option(c("-m", "--method"))
)

opt <- parse_args(OptionParser(option_list=option_list))

df_anno <- read.table("../../Statistics/stat.txt", header = TRUE)
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


method <- opt$method
if (method == "Raw"){
        filename <- "../../TagCount/TagCount.txt"
    } else if (method == "scImpute"){
        filename <- "../../Imputation/scImpute/scimpute_count.txt"
    } else if (method == "SCALE"){
        filename <- "../../Imputation/SCALE/imputed_data.txt"
    } else if (method == "DCA"){
        filename <- "../..Imputation/DCA/mean.tsv"
    } else{
        filename <- sprintf("../../Imputation/%s/%s.txt",
                            method, method)
    }
    mat <- read.table(filename, header = TRUE)
    df_anno <- df_anno[colnames(mat), ]
    dist = 1 - cor(mat)
    
    si <- silhouette(x = df_anno$CellType, 
                     dmatrix = dist)
    
    saveRDS(si, sprintf("../%s.Rds", method))
    
    pdf(sprintf("../%s.pdf", method), height = 8, width = 8)
    plot(si)
    dev.off()