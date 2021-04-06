library(fclust)
library(cluster)
library(e1071)
library(foreach)
library(optparse)
library(Rtsne)
library(patchwork)

option_list <- list( 
    make_option(c("-i", "--input"),
                help="input filename"),
    make_option(c("-o", "--output_dir"), 
                help="output filename"),
    make_option(c("-n", "--num_clusters"), 
                help="number of clusters"),
    make_option(c("-m", "--method"), 
                help="number of clusters")
)

opt <- parse_args(OptionParser(option_list=option_list))

set.seed(42)

get_pca <- function(mat){
    mat <- as.matrix(mat)
    mat <- 1e5 * sweep(mat, 2, colSums(mat), "/")
    mat.pca <- prcomp(t(mat), rank. = 50)
    x <- as.data.frame(mat.pca$x)
    return(x)
}

run_kmedoids <- function(df.dist, num_clusters){
    pa <- pam(df.dist, 
              k = num_clusters, 
              diss = TRUE, 
              cluster.only = TRUE)
    df_cluster <- as.data.frame(pa)
    
    return(df_cluster)
}

run_hc <- function(df.dist, num_clusters){
    hc <- hclust(d = df.dist, method = "ward.D2")
    memb <- cutree(hc, k = num_clusters)
    df_cluster <- as.data.frame(memb)

    return(df_cluster)
}

set.seed(42)

stopifnot(file.exists(opt$input))
df <- read.table(opt$input, header = TRUE)

df.pca <- get_pca(df)
df.dist.pca <- as.dist(1 - cor(t(df.pca)))


tsne_out <- Rtsne(df.pca, pca = FALSE, verbose = FALSE)
df_tsne_out <- as.data.frame(tsne_out$Y)
colnames(df_tsne_out) <- c("tSNE1", "tSNE2")
df.dist.tsne <- dist(as.matrix(df_tsne_out), 
                     method = "euclidean")


if(opt$method == "kmedoids"){
    output_filename <- sprintf("%s/kmedoids_pca_%s.txt", 
                               opt$output_dir, 
                               opt$num_clusters)
    
    df_cluster <- run_kmedoids(df.dist = df.dist.pca,
                               num_clusters = opt$num_clusters)
    
    write.table(df_cluster, output_filename, quote = FALSE, sep = "\t")
    
    
    output_filename <- sprintf("%s/kmedoids_tsne_%s.txt", 
                               opt$output_dir, 
                               opt$num_clusters)
    
    df_cluster <- run_kmedoids(df.dist = df.dist.tsne,
                               num_clusters = opt$num_clusters)
    rownames(df_cluster) <- colnames(df)
    
    write.table(df_cluster, output_filename, quote = FALSE, sep = "\t")
} else if(opt$method == "hc"){
    output_filename <- sprintf("%s/hc_pca_%s.txt", 
                               opt$output_dir, 
                               opt$num_clusters)
    
    df_cluster <- run_hc(df.dist = df.dist.pca,
                         num_clusters = opt$num_clusters)
    
    write.table(df_cluster, output_filename, quote = FALSE, sep = "\t")
    
    
    output_filename <- sprintf("%s/hc_tsne_%s.txt", 
                               opt$output_dir, 
                               opt$num_clusters)
    
    df_cluster <- run_hc(df.dist = df.dist.tsne,
                         num_clusters = opt$num_clusters)
    rownames(df_cluster) <- colnames(df)
    write.table(df_cluster, output_filename, quote = FALSE, sep = "\t")
}
