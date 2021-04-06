library(ggplot2)
library(cowplot)
library(gridExtra)
library(dplyr)
library(mclust)
library(ComplexHeatmap)
library(RColorBrewer)

df_tagcount <- read.table("TagCount_Cor.txt")
df_BNMF <- read.table("BNMF_Cor.txt")

rownames(df_tagcount) <- df_tagcount$V1
rownames(df_BNMF) <- df_BNMF$V1

df_tagcount$V1 <- NULL
df_BNMF$V1 <- NULL

df_tagcount <- as.data.frame(t(apply(df_tagcount, 1, FUN = function(X) (X - min(X))/diff(range(X))))) 
df_BNMF <- as.data.frame(t(apply(df_BNMF, 1, FUN = function(X) (X - min(X))/diff(range(X)))))

colnames(df_tagcount) <- c("1", "2", "3", "4", "5", "6")
colnames(df_BNMF) <- c("1", "2", "3", "4", "5", "6")

df_anno <- read.table("../../Statistics/stat.txt", header = TRUE)
df_anno <- subset(df_anno, Runs %in% rownames(df_tagcount))
df_anno$CellType <- factor(df_anno$CellType, 
                           levels = c("BJ", "GM12878", "H1-ESC", "HL60", "K562", "TF1"))


df_tagcount_cluster <- read.table("./TagCount.txt", header = TRUE)
df_tagcount_cluster$LandMarkAssignments <- as.factor(df_tagcount_cluster$LandMarkAssignments)
df_tagcount_cluster$Runs <- rownames(df_tagcount_cluster)

df_atacImpute_cluster <- read.table("./BNMF.txt", header = TRUE)
df_atacImpute_cluster$LandMarkAssignments <- as.factor(df_atacImpute_cluster$LandMarkAssignments)
df_atacImpute_cluster$Runs <- rownames(df_atacImpute_cluster)

colnames(df_tagcount_cluster) <- c("ClusterTC", "Runs")
colnames(df_atacImpute_cluster) <- c("ClusteratacImpute", "Runs")

df_anno <- merge(df_anno, df_tagcount_cluster, by = "Runs")
df_anno <- merge(df_anno, df_atacImpute_cluster, by = "Runs")

df_anno <- df_anno[order(df_anno$CellType), ]

cols <- colorRampPalette(c("white", "red"), space = "rgb")(256)
col_list1 <- list(CellType = c("H1-ESC" = "#a6cee3", "BJ" = "#1f78b4", 
                              "GM12878" = "#b2df8a", "K562" = "#33a02c",
                              "HL60" = "#fb9a99", "TF1" = "#e31a1c"))

col_list2 <- list(Cluster = c("1" = "#7fc97f", "2" = "#beaed4", 
                               "3" = "#fdc086", "4" = "#ffff99",
                               "5" = "#bf5b17", "6" = "#f0027f"))

df_tagcount <- df_tagcount[df_anno$Runs, ]
df_BNMF <- df_BNMF[df_anno$Runs, ]
df_tagcount <- df_tagcount[c("5", "2", "1", "4", "6", "3")]
df_BNMF <- df_BNMF[c("5", "2", "1", "4", "6", "3")]

colnames(df_tagcount) <- c("1", "2", "3", "4", "5", "6")
colnames(df_BNMF) <- c("1", "2", "3", "4", "5", "6")

ha1 <- HeatmapAnnotation(CellType = df_anno$CellType, col = col_list1, which = "row")

ha_cluster_tc <- HeatmapAnnotation(Cluster = df_anno$ClusterTC, col = col_list2, which = "row")

ha_cluster_atacImpute <- HeatmapAnnotation(Cluster = df_anno$ClusteratacImpute, col = col_list2, which = "row")

p1 <- ha1 + ha_cluster_tc + Heatmap(matrix = as.matrix(df_tagcount),
                  name = "Similarity",
                  show_row_dend = FALSE,
                  show_column_dend = FALSE,
                  cluster_rows = FALSE,
                  cluster_columns = FALSE,
                  show_row_names = FALSE,
                  col = cols)

p2 <- ha1 + ha_cluster_atacImpute + Heatmap(matrix = as.matrix(df_BNMF),
                  name = "Similarity",
                  show_row_dend = FALSE,
                  show_column_dend = FALSE,
                  cluster_rows = FALSE,
                  cluster_columns = FALSE,
                  show_row_names = FALSE,
                  col = cols)

pdf("TagCount_Cor.pdf", height = 4, width = 6)
print(p1)
dev.off()

pdf("BNMF_Cor.pdf", height = 4, width = 6)
print(p2)
dev.off()