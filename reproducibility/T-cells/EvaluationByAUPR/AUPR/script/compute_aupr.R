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

df_true <- read.table("../../TrueLabels/true_labels.txt",
                      header = TRUE)
stopifnot(file.exists(opt$input))
df_score <- read.table(opt$input, header = TRUE)

##########################################################################
# test
# df_score <- read.table("../../../TagCount/TagCount.txt", header = TRUE)
# df_score <- read.table("../../../Imputation/SCALE/SCALE.txt", header = TRUE)
##############################################################################

df_anno <- read.table("../../../Statistics/stat.txt", header = TRUE)
df_anno <- subset(df_anno, Runs %in% colnames(df_score))
celltype_list <- unique(df_anno$CellType)

df_auc <- setNames(data.frame(matrix(ncol = 3, 
                                     nrow = ncol(df_score))), 
                   c("Runs", "CellType", "AUPR"))
idx <- 1
for(celltype in celltype_list){
    message(sprintf("computing AUPR for cell type: %s \n", celltype))
    df_anno_sub <- subset(df_anno, CellType == celltype)

    for(cell in df_anno_sub$Runs){
        pr <- pr.curve(scores.class0 = df_score[[cell]][df_true[[celltype]] == 1],
                       scores.class1 = df_score[[cell]][df_true[[celltype]] == 0],
                       curve = FALSE)
        df_auc[["Runs"]][idx] <- cell
        df_auc[["CellType"]][idx] <- celltype
        df_auc[["AUPR"]][idx] <- round(pr$auc.integral, digits = 3)
        idx <- idx + 1
    }
    
}
write.table(df_auc, opt$output, quote = FALSE, sep = "\t", row.names = FALSE)
