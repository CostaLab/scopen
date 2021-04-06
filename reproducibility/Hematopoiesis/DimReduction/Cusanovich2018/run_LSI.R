library(Signac)
library(Seurat)
library(dplyr)
library(Matrix)
library(irlba)
library(SnapATAC)

# https://shendurelab.github.io/fly-atac/docs/#usecase2

# read binary matrix
x.sp <- readRDS("../SnapATAC/x.sp.Rds")
counts <- t(x.sp@bmat)
rownames(counts) <- x.sp@feature$name
counts <- counts[sort(rowSums(counts), 
                 index=T, 
                 decreasing=TRUE)$ix, ]


# select top 20,000 windows
counts <- counts[c(1:20000), ]

chrom_assay <- CreateChromatinAssay(
    counts = counts,
    sep = c(":", "-"),
    genome = 'hg38',
    min.cells = 1,
    min.features = 1
)

obj <- CreateSeuratObject(
    counts = chrom_assay,
    assay = "peaks"
)

obj <- RunTFIDF(obj, method = 2)
obj <- FindTopFeatures(obj, min.cutoff = 'q0')
obj <- RunSVD(obj)

saveRDS(obj, file = "obj.Rds")
