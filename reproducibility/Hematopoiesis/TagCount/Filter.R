library(ggplot2)

df <- read.table("TagCount.txt", header = TRUE)
df2 <- read.table("../Imputation/BNMF/BNMF.txt", header = TRUE)

peaks <- rownames(df2)
cells <- colnames(df2)

df3 <- df[peaks, cells]


write.table(df3, file = "TagCount_filter.txt", quote = FALSE, sep = "\t")
