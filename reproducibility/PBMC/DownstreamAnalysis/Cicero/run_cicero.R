library(ggplot2)
library(cicero)
library(reshape2)
library(optparse)
library(scater)
library(fclust)
library(cluster)
library(e1071)
library(dplyr)
library(mclust)
library(Rtsne)
library(cowplot)
library(gridExtra)

num_clusters <- 14
data_list <- c("scOpen")

for (data in data_list) {
    if(data == "Raw"){
        input_file <- "../../TagCount/TagCount.txt"
    } else{
        input_file <- "../../Imputation/scOpen/scOpen.txt"
    }
    
    if(!dir.exists(data)){
        dir.create(data)
    }
    
    df <- read.table(input_file, header = TRUE)
    df_anno <- read.table("../../Statistics/stat.txt", header = TRUE) %>%
        subset(., Cell %in% colnames(df))

    df$Peak <- rownames(df)
    df <- melt(df)
    
    colnames(df) <- c("Peak", "Cell", "Count")
    df <- subset(df, Count > 0.001)
 
    df <- droplevels(df)
    input_cds <- make_atac_cds(df)
    input_cds <- detectGenes(input_cds)
    input_cds <- estimateSizeFactors(input_cds)
    
    input_cds <- reduceDimension(input_cds, max_components = 2, 
                                 norm_method = 'none',
                                 num_dim = 6, reduction_method = 'tSNE',
                                 verbose = TRUE, check_duplicates = FALSE,
                                 pseudo_expr = 0)
    
    tsne_coords <- t(reducedDimA(input_cds))
    row.names(tsne_coords) <- row.names(pData(input_cds))
    cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = tsne_coords)
    
    chrom_size <- read.table("chrom.sizes.hg38")
    conns <- run_cicero(cds = cicero_cds, 
                        genomic_coords = chrom_size) # Takes a few minutes to run
    
    gene_annotation <- read.table("hg38.annotation.bed", header = TRUE)
    gene_annotation_sub <- gene_annotation[,c(1:3, 8)]
    names(gene_annotation_sub)[4] <- "gene"
    
    input_cds <- annotate_cds_by_site(input_cds, gene_annotation_sub, verbose = TRUE)
    unnorm_ga <- build_gene_activity_matrix(input_cds, conns)
    
    # remove any rows/columns with all zeroes
    unnorm_ga <- as.data.frame(as.matrix(unnorm_ga[!Matrix::rowSums(unnorm_ga) == 0, ]))

    write.table(unnorm_ga, 
                file = sprintf("./%s/GA.txt", data), 
                quote = FALSE, 
                sep = "\t")
    
    df.pca <- prcomp(t(unnorm_ga), 
                     rank. = 50, 
                     center = TRUE, 
                     scale. = TRUE)
    
    df.dist <- as.dist(1 - cor(t(df.pca$x)))
    pa <- pam(df.dist, k = num_clusters, diss = TRUE, cluster.only = TRUE)
    df_cluster <- as.data.frame(pa)
    
    df_cluster$Cell <- rownames(df_cluster)
    colnames(df_cluster) <- c("Cluster", "Cell")
    
    df_cluster <- merge(df_cluster, df_anno, by = "Cell")
    write.table(df_cluster, file = sprintf("./%s/clustering.txt", data), 
                quote = FALSE, sep = "\t", row.names = FALSE)
    
    df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$Cluster, df_cluster$CellType))
    colnames(df_ARI) <- "ARI"
    write.table(df_ARI, 
                file = sprintf("./%s/ARI.txt", data),
                quote = FALSE, sep = "\t", 
                row.names = FALSE)
    
    tsne_out <- Rtsne(df.dist, is_distance = TRUE) # Run TSNE
    df_plot <- as.data.frame(tsne_out$Y)
    colnames(df_plot) <- c("tSNE1", "tSNE2")
    df_plot$Cell <- rownames(df.pca$x)
    df_plot <- merge(df_plot, df_anno, by = "Cell")
    
    p <- ggplot(data = df_plot, aes(x = tSNE1, y = tSNE2, color = CellType)) +
        geom_point() +
        theme_cowplot() +
        theme(legend.title = element_blank())
    
    pdf(file = sprintf("./%s/tsne.pdf", data), height = 8, width = 8)
    print(p)
    dev.off()
    
    write.table(df_plot, 
                file = sprintf("./%s/tsne.txt", data), 
                quote = FALSE, sep = "\t", row.names = FALSE)
}







