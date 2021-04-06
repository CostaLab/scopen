library(fclust)
library(cluster)
library(e1071)
library(foreach)
library(optparse)
library(PRROC)

option_list <- list( 
    make_option(c("-i", "--input"),
                help="input filename"),
    make_option(c("-o", "--output"), 
                help="output filename")
)
opt <- parse_args(OptionParser(option_list=option_list))
stopifnot(file.exists(opt$input))
df_score <- read.table(opt$input, header = TRUE)

print(dim(df_score))

df_true <- read.table("../../TrueLabels/true_labels.txt",
                      header = TRUE)

df_anno <- read.table("../../../Statistics/stat.txt", header = TRUE, sep = "\t")
df_anno$CellType <- stringr::str_replace_all(df_anno$CellType, " ", "_")
df_anno$CellType <- stringr::str_replace_all(df_anno$CellType, 
                                             c("CD56_\\(dim\\)_NK_cells" = "CD56_.dim._NK_cells",
                                               "CD56_\\(bright\\)_NK_cells" = "CD56_.bright._NK_cells",
                                               "non-classical_monocytes" = "non.classical_monocytes"))

df_auc <- setNames(data.frame(matrix(ncol = 3,
                                     nrow = ncol(df_score))),
                   c("Cells", "CellType", "AUPR"))
idx <- 1
for(celltype in unique(df_anno$CellType)){
    message(sprintf("computing AUPR for cell type: %s \n", celltype))
    df_anno_sub <- subset(df_anno, CellType == celltype)

    for(cell in df_anno_sub$Cell){
        pr <- pr.curve(scores.class0 = df_score[[cell]][df_true[[celltype]] == 1],
                       scores.class1 = df_score[[cell]][df_true[[celltype]] == 0],
                       curve = FALSE)
        df_auc[["Cells"]][idx] <- cell
        df_auc[["CellType"]][idx] <- celltype
        df_auc[["AUPR"]][idx] <- round(pr$auc.integral, digits = 3)
        idx <- idx + 1
    }
    
}
write.table(df_auc, opt$output, quote = FALSE, sep = "\t", row.names = FALSE)
