library(SnapATAC)
library(GenomicRanges)
library(dplyr)
library(mclust)
library(cluster)
library(ArchR)

sample_list <- c("D0_1", "D0_2",
                 "D2_1", "D2_2",
                 "D10_1", "D10_2")
# Select barcode
proj <- loadArchRProject(path = "../../ArchR/UUO")
df <- proj@cellColData
barcode.list <- stringr::str_split_fixed(rownames(df), "#", 2) %>%
    as.data.frame()
colnames(barcode.list) <- c("sample", "barcode")
barcode.list <- split(barcode.list$barcode, barcode.list$sample)

x.sp.list <- lapply(sample_list, function(x){
    x.sp <- createSnap(file = sprintf("./%s.snap", x), 
                      sample=x)
    x.sp  = x.sp[x.sp@barcode %in% barcode.list[[x]], ]
    x.sp <- addBmatToSnap(x.sp, bin.size=5000, 
                         num.cores=10)
    
    x.sp
})

bin.shared = Reduce(intersect, lapply(x.sp.list, function(x.sp) x.sp@feature$name))
x.sp.list <- lapply(x.sp.list, function(x.sp){
    idy = match(bin.shared, x.sp@feature$name);
    x.sp[,idy, mat="bmat"];
})
x.sp = Reduce(snapRbind, x.sp.list)
rm(x.sp.list)
gc()
table(x.sp@sample)

# Binarize matrix
x.sp <- makeBinary(x.sp, mat="bmat")


# Filter bins
# First, we filter out any bins overlapping with the ENCODE blacklist to prevent from potential artifacts.
if(!file.exists("./whg38.blacklist.bed.gz")){
    system("wget http://mitra.stanford.edu/kundaje/akundaje/release/blacklists/hg38-human/hg38.blacklist.bed.gz")
}

black_list = read.table('hg38.blacklist.bed.gz')
black_list.gr = GRanges(black_list[,1], 
                        IRanges(black_list[,2], black_list[,3]))
idy = queryHits(findOverlaps(x.sp@feature, black_list.gr))
if(length(idy) > 0){
    x.sp = x.sp[,-idy, mat="bmat"]
    }

# Second, we remove unwanted chromosomes.
chr.exclude = seqlevels(x.sp@feature)[grep("random|chrM", seqlevels(x.sp@feature))]
idy = grep(paste(chr.exclude, collapse="|"), x.sp@feature)
if(length(idy) > 0){x.sp = x.sp[,-idy, mat="bmat"]}

# Third, the bin coverage roughly obeys a log normal distribution. We remove the top 5% bins that overlap with invariant features such as promoters of the house keeping genes.
bin.cov = log10(Matrix::colSums(x.sp@bmat)+1)
bin.cutoff = quantile(bin.cov[bin.cov > 0], 0.95)
idy = which(bin.cov <= bin.cutoff & bin.cov > 0)
x.sp = x.sp[, idy, mat="bmat"]

# Reduce dimensionality
row.covs = log10(Matrix::rowSums(x.sp@bmat)+1)
row.covs.dens = density(
    x = row.covs, 
    bw = 'nrd', adjust = 1
)
sampling_prob = 1 / (approx(x = row.covs.dens$x, y = row.covs.dens$y, xout = row.covs)$y + .Machine$double.eps)
set.seed(1)
idx.landmark.ds = sort(sample(x = seq(nrow(x.sp)), size = 10000, prob = sampling_prob))
x.landmark.sp = x.sp[idx.landmark.ds,]
x.query.sp = x.sp[-idx.landmark.ds,]
x.landmark.sp = runDiffusionMaps(
    obj= x.landmark.sp,
    input.mat="bmat", 
    num.eigs=50
)

x.query.sp = runDiffusionMapsExtension(
    obj1=x.landmark.sp, 
    obj2=x.query.sp,
    input.mat="bmat"
)
x.landmark.sp@metaData$landmark = 1
x.query.sp@metaData$landmark = 0
x.sp = snapRbind(x.landmark.sp, x.query.sp)
x.sp = x.sp[order(x.sp@sample),]
rm(x.landmark.sp, x.query.sp)

pdf("plotDimReductPW.pdf")
plotDimReductPW(
    obj=x.sp, 
    eigs.dims=1:50,
    point.size=0.3,
    point.color="grey",
    point.shape=19,
    point.alpha=0.6,
    down.sample=5000,
    pdf.file.name=NULL, 
    pdf.height=7, 
    pdf.width=7
)
dev.off()


saveRDS(x.sp, "x.sp.Rds")