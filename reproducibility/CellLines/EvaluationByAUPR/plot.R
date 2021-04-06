library(ggplot2)
library(cowplot)
library(gridExtra)
library(dplyr)
library(tidyr)
library(ComplexHeatmap)
library(reshape2)
library(viridis)
library(PMCMRplus)
library(openxlsx)

method_list <- c('scImpute','MAGIC', 'SAVER', 'cisTopic',
                 'DCA', 'scBFA', 'SCALE', 'Raw', 'PCA',
                 'scOpen')

df_list <- vector(mode = "list")
idx <- 1
for(method in method_list){
    df_list[[idx]] <- read.table(sprintf("./AUPR/%s.txt", method), header = TRUE)
    df_list[[idx]]$Method <- method
    idx <- idx + 1
}

df <- Reduce(rbind, df_list)

cols <- rainbow(length(method_list), s=0.8, v=0.9)

p <- ggplot(df, aes(x = reorder(Method, -AUPR, FUN = median), y = AUPR)) +
    geom_boxplot(aes(fill = Method)) +
    scale_fill_manual(values = cols) +
    xlab("") + ylab("") +
    theme_cowplot() +
    theme(axis.text.x = element_text(angle = 60, hjust = 1),
          legend.title = element_blank(),
          legend.position = "none")

pdf(file = "AUPR.pdf", height = 6, width = 10)
print(p)
dev.off()
