library(ggplot2)
library(cowplot)
library(gridExtra)
library(cluster)
library(fclust)
library(PMCMRplus)
library(openxlsx)
library(dplyr)
library(tidyr)
library(randomcoloR)

df_anno <- read.table("../Statistics/stat.txt", header = TRUE)
df_anno$NumofReads <- NULL

method_list <- c('scImpute','MAGIC', 'SAVER', 'cisTopic',
                 'DCA', 'scBFA', 'SCALE', 'Raw', 'PCA',
                 'scOpen')

ARI_list <- vector(mode = "list")
idx <- 1

for(num_clusters in c(6, 7)){
    for(input in c("pca", "tsne")){
        for(clustering_method in c("hc", "kmedoids")){
            for(method in method_list){
                df <- read.table(file = sprintf("k_%s/%s/%s_%s_%s.txt", num_clusters,
                                                method, clustering_method, input,
                                                num_clusters))
                colnames(df) <- c("Cluster")
                df$Runs <- rownames(df)
                df_anno <- subset(df_anno, Runs %in% df$Runs)
                df <- merge(df, df_anno, by = "Runs")
                
                ARI <- mclust::adjustedRandIndex(df$Cluster, df$CellType)
                ARI_list[[idx]] <- data.frame(ARI, 
                                              num_clusters, 
                                              clustering_method, 
                                              method,
                                              input)
                idx <- idx + 1
            }
        }
        
    }
}

df_ARI <- Reduce(rbind, ARI_list)

cols <- rainbow(length(method_list), s=0.8, v=0.9)

p <- ggplot(data = df_ARI, aes(x = method, y = ARI, fill = method)) +
    geom_bar(position = "dodge", stat = "identity") +
    facet_wrap(~num_clusters + clustering_method + input, ncol = 4) +
    scale_fill_manual(values = cols) +
    xlab("") +
    theme_cowplot() +
    theme(legend.title = element_blank(),
          legend.position = "none",
          axis.text.x = element_text(angle = 90, hjust = 1))


pdf("ARI.pdf", width = 18, height = 10)
print(p)
dev.off()

df_ARI_rank <- df_ARI %>%
    group_by(clustering_method, num_clusters, input) %>%
    mutate(rank = order(order(ARI, decreasing=TRUE))) %>%
    ungroup()


p <- ggplot(data = df_ARI_rank, aes(x = reorder(method, rank, median),
                                    y = rank)) +
    geom_boxplot(aes(fill = method)) +
    scale_y_continuous(breaks = c(1, 3, 5, 7, 9, 11)) +
    scale_fill_manual(values = cols) +
    theme_cowplot() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 90, hjust = 1))

pdf("ARI_all.pdf", width = 8, height = 6)
print(p)
dev.off()

write.table(df_ARI, file = "ARI.txt", sep = "\t", quote = FALSE,
            row.names = FALSE)
