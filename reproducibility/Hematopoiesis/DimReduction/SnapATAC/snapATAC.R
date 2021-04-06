library(SnapATAC)
library(GenomicRanges)
library(dplyr)
library(mclust)
library(cluster)

x.sp = createSnap(
    file="fragment.snap",
    sample="T-cells",
    do.par = TRUE,
    num.cores=10)

x.sp = addBmatToSnap(x.sp, bin.size=5000, num.cores=10)

x.sp = makeBinary(x.sp, mat="bmat")


if(!file.exists("./wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz")){
    system("wget http://mitra.stanford.edu/kundaje/akundaje/release/blacklists/hg19-human/wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz")
}

black_list = read.table('wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz')
black_list.gr = GRanges(black_list[,1], 
                        IRanges(black_list[,2], black_list[,3]))
idy1 = queryHits(findOverlaps(x.sp@feature, black_list.gr))
idy2 = grep("chrM|random", x.sp@feature)
idy = unique(c(idy1, idy2))
x.sp = x.sp[,-idy, mat="bmat"]

x.sp = filterBins(
    x.sp,
    low.threshold=-2,
    high.threshold=2,
    mat="bmat"
)

x.sp = runDiffusionMaps(
    obj=x.sp,
    input.mat="bmat", 
    num.eigs=50
)

saveRDS(x.sp, "x.sp.Rds")


#######################################################
# clustering with scOpen dimension reduction matrix
x.sp = runKNN(
    obj=x.sp,
    eigs.dims=1:20,
    k=15
)

x.sp=runCluster(
    obj=x.sp,
    tmp.folder=tempdir(),
    louvain.lib="R-igraph",
    seed.use=10
)
x.sp@metaData$Cluster = x.sp@cluster

df_cluster <- as.data.frame(x.sp@metaData)

df_anno <- read.table("../Statistics/stat.txt", header = TRUE) %>%
    subset(., Runs %in% df_cluster$barcode) %>%
    rename(., barcode = Runs)

df_cluster <- merge.data.frame(df_cluster, df_anno)
df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType,
                                          df_cluster$Cluster))
colnames(df_ARI) <- "ARI"
dir.create("./Raw")

write.table(df_ARI, file = "./Raw/ARI.txt",
            quote = FALSE, sep = "\t", row.names = FALSE)


#######################################################
# clustering with scOpen dimension reduction matrix
# we use code from https://github.com/r3fang/SnapATAC/blob/master/R/runKNN.R
# and https://github.com/r3fang/SnapATAC/blob/master/R/runClusters.R
df <- read.table("../Imputation/scOpen/scOpen_barcodes.txt", 
                 header = TRUE)
x.sp@smat@dmat <- as.matrix(t(df))

library(matrixStats)

x.sp@smat@sdev <- rowSds(as.matrix(df))
x.sp = runKNN(
    obj=x.sp,
    eigs.dims=1:nrow(df),
    k=15
)
x.sp=runCluster(
    obj=x.sp,
    tmp.folder=tempdir(),
    louvain.lib="R-igraph",
    seed.use=10
)

x.sp@metaData$Cluster = x.sp@cluster

df_cluster <- as.data.frame(x.sp@metaData)
df_cluster$barcode <- colnames(df)

df_cluster <- merge.data.frame(df_cluster, df_anno)
df_ARI <- as.data.frame(adjustedRandIndex(df_cluster$CellType,
                                          df_cluster$Cluster))
colnames(df_ARI) <- "ARI"

dir.create("./scOpen")

write.table(df_ARI, file = "./scOpen/ARI.txt",
            quote = FALSE, sep = "\t", row.names = FALSE)
