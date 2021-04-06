library(BiocParallel)
register(MulticoreParam(48)) # Use 48 cores
library(chromVAR)
library(mclust)
library(motifmatchr)
library(BSgenome.Hsapiens.UCSC.hg19)
library(stringr)
library(GenomicRanges)
library(JASPAR2020)
library(IRanges)
library(SummarizedExperiment)
library(TFBSTools)
library(pheatmap)
library(pdfCluster)
library(ggplot2)
library(cowplot)
library(gridExtra)
library(e1071)
library(dplyr)
library(tibble)
library(Seurat)
library(cowplot)
library(gridExtra)
library(glue)

num_clusters <- 8

col_list <- list(CellType = c("CLP" = "#98D9E9", "CMP" = "#FFC179", 
                              "GMP" = "#FFA300", "HSC" = "#00441B",
                              "LMPP" = "#00AF99", "MEP" = "#F6313E",
                              "MPP" = "#46A040", "pDC" = "#C390D4"))

cols <- c("CLP" = "#98D9E9", "CMP" = "#FFC179", 
          "GMP" = "#FFA300", "HSC" = "#00441B",
          "LMPP" = "#00AF99", "MEP" = "#F6313E",
          "MPP" = "#46A040", "pDC" = "#C390D4")

data_list <- c("Raw", "scOpen")

for(data in data_list){
    if(data == "Raw"){
        input_file <- "../../TagCount/TagCount.txt"
    } else{
        input_file <- "../../Imputation/scOpen/scOpen.txt"
    }
    
    if(!dir.exists(data)){
        dir.create(data)
    }
    
    df <- read.table(input_file, header = TRUE)

    df_anno <- read.table(glue("../../Visualization/UMAP/{data}.txt"),
                          header = TRUE)
    rownames(df_anno) <- df_anno$Runs
    df_anno <- df_anno[colnames(df), ]
    
    coord_cols <- str_split_fixed(rownames(df), ":|-|_", 3)
    rowRanges <- GRanges(coord_cols[, 1],
                         ranges = IRanges(as.numeric(coord_cols[, 2]), 
                                          as.numeric(coord_cols[, 3])))
    
    rse <- SummarizedExperiment(assays = SimpleList(counts=as.matrix(df)),
                                rowRanges = rowRanges, 
                                colData = df_anno[, c("CellType")])
    
    counts <- addGCBias(rse, genome = BSgenome.Hsapiens.UCSC.hg19)

    opts <- list()
    opts["species"] <- "Homo sapiens"
    opts["collection"] <- "CORE"
    motifs1 <- getMatrixSet(JASPAR2020, opts)
    
    opts["species"] <- "Mus musculus"
    opts["collection"] <- "CORE"
    motifs2 <- getMatrixSet(JASPAR2020, opts)
    
    names(motifs1) <- paste(names(motifs1), name(motifs1), sep = "_")
    names(motifs2) <- paste(names(motifs2), name(motifs2), sep = "_")
    
    motifs1 <- motifs1[!grepl("var.", name(motifs1))]
    motifs2 <- motifs2[!grepl("var.", name(motifs2))]
    
    motif_names <- unique(c(names(motifs1), names(motifs2)))
    
    motifs <- c(motifs1, motifs2)
    motifs <- motifs[motif_names, ]
    
    motif_ix <- matchMotifs(motifs, counts,
                            genome = BSgenome.Hsapiens.UCSC.hg19)
    
    # computing deviations
    dev <- computeDeviations(object = counts, 
                             annotations = motif_ix)
    
    saveRDS(dev, file = sprintf("./%s/chromVAR.rds", data))
    
    # Clustering
    sample_cor <- getSampleCorrelation(dev,
                                       threshold = 0)
    sample_cor[is.na(sample_cor)] <- 0
   
    df_anno2 <- subset(df_anno, select = c("CellType"))
    tiff(file = sprintf("./%s/heatmap.tiff", data), width = 8, height = 8,
         units = "in", type = "cairo", res = 300)
    out <- pheatmap(as.dist(sample_cor), 
                    annotation_row = df_anno2, 
                    clustering_distance_rows = as.dist(1-sample_cor), 
                    clustering_distance_cols = as.dist(1-sample_cor),
                    annotation_colors = col_list)
    dev.off()
    
    df_cluster <- as.data.frame(sort(cutree(out$tree_row, 
                                            k = num_clusters)))
    colnames(df_cluster) <- c("Cluster")
    
    df_cluster$Cell <- rownames(df_cluster)
    df_anno$Cell <- rownames(df_anno)
    
    df_cluster <- merge.data.frame(df_cluster,
                                   df_anno,
                                   by = "Cell")
    
    df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType, 
                                              df_cluster$Cluster))
    colnames(df_ARI) <- "ARI"
    
    write.table(df_cluster, file = sprintf("./%s/clustering.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)
    
    write.table(df_ARI, file = sprintf("./%s/ARI.txt", data),
                quote = FALSE, sep = "\t", row.names = FALSE)

    
    df_dev <-as.data.frame(dev@assays@data[['z']])
    
    df_dev[is.na(df_dev)] <- 0
    write.table(df_dev, file = sprintf("./%s/dev.txt", data), 
                quote = FALSE, sep = "\t")
    
    
    obj <- CreateSeuratObject(counts = df_dev, 
                                      min.cells = 0, 
                                      min.features = 0, 
                                      names.field = 1, names.delim = "_")
    
    obj <- AddMetaData(obj, df_anno$CellType, col.name = "CellType")
    
    all.tfs <- rownames(obj)
    #obj <- ScaleData(obj, features = all.tfs)
    
    embeddings <- as.matrix(df_anno[, c("UMAP1", "UMAP2")])
    
    obj[['umap']] <- CreateDimReducObject(embeddings = embeddings,
                                          key = "UMAP",
                                          assay = "RNA")
    
    if(!dir.exists(glue("./{data}/VlnPlotByCellType"))){
        dir.create(glue("./{data}/VlnPlotByCellType"))
    }
    
    for (tf in all.tfs) {
        p <- VlnPlot(obj, features = tf, group.by = "CellType",
                     pt.size = 0.0, combine = TRUE, cols = cols) + 
            xlab("") + ylab("Z-score")
        pdf(glue("./{data}/VlnPlotByCellType/{tf}.pdf"), 
            height = 6, width = 6)
        print(p)
        dev.off()
    }
    
    if(!dir.exists(glue("./{data}/UMAP"))){
        dir.create(glue("./{data}/UMAP"))
    }
    
    for (tf in all.tfs) {
        p <- FeaturePlot(obj, reduction = "umap", features = tf)+
            scale_color_gradient2(midpoint=0, low="blue", mid="gray",
                                  high="red", space ="Lab" )

        pdf(glue("./{data}/UMAP/{tf}.pdf"), 
            height = 6, width = 6)
        print(p)
        dev.off()
    }
}



