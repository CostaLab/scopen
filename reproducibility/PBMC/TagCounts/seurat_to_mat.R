library(Seurat)
library(Matrix)
library(dplyr)

# https://raw.githack.com/bioFAM/MOFA2/master/MOFA2/vignettes/10x_scRNA_scATAC.html
# run this to download the data set which includes cell annoation
filename <- "./seurat.rds"
if(file.exists(filename)){
    pbmc <- readRDS(filename)
} else{
    pbmc <- readRDS(url("ftp://ftp.ebi.ac.uk/pub/databases/mofa/10x_rna_atac_vignette/seurat.rds"))
    saveRDS(pbmc, filename)
}

DefaultAssay(pbmc) <- "ATAC"

# Keep cells that pass QC for both omics
pbmc <- pbmc %>%
    .[, pbmc@meta.data$pass_accQC==TRUE & 
          pbmc@meta.data$pass_rnaQC==TRUE]

counts <- as.matrix(pbmc@assays$ATAC@counts)

# only keep features that are found in at least 50 cells
counts <- counts[rowSums(counts > 0) > 3, ]

rownames(counts) <- stringr::str_replace_all(rownames(counts), ":", "_")
rownames(counts) <- stringr::str_replace_all(rownames(counts), "-", "_")

chr <- stringr::str_split_fixed(rownames(counts), "_", 3)[, 1]

counts <- counts[chr %in% paste0("chr",c(1:22,"X")), ]

write.table(counts, file = "TagCount.txt", quote = FALSE, sep = "\t")

