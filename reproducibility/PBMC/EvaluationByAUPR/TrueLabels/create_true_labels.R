library(ggplot2)
library(cowplot)
library(gridExtra)
library(GenomicRanges)
library(stringr)

df <- read.table("../../TagCount/TagCount.txt", header = TRUE)

chrom <- str_split_fixed(rownames(df), "_", n = 3 )[, 1]
start <- as.numeric(str_split_fixed(rownames(df), "_", n = 3 )[, 2])
end <- as.numeric(str_split_fixed(rownames(df), "_", n = 3 )[, 3])
peaks_all <- GRanges(seqnames = chrom, ranges = IRanges(start = start, end = end))

print(head(peaks_all))

celltype_list <- c("CD56_(bright)_NK_cells",
                   "CD56_(dim)_NK_cells",
                   "classical_monocytes",
                   "effector_CD8_T_cells",
                   "intermediate_monocytes",
                   "MAIT_T_cells",
                   "memory_B_cells",
                   "memory_CD4_T_cells",
                   "myeloid_DC",
                   "naive_B_cells",
                   "naive_CD4_T_cells",
                   "naive_CD8_T_cells",
                   "non-classical_monocytes",
                   "plasmacytoid_DC")

df_true <- data.frame(matrix(ncol = length(celltype_list), nrow = nrow(df)))
colnames(df_true) <- celltype_list
rownames(df_true) <- rownames(df)

for(celltype in celltype_list){
    df_peaks <- read.table(sprintf("../CellTypePeaks/Peaks/%s_peaks.narrowPeak",
                                   celltype))
    df_peaks <- subset(df_peaks, select = c("V1", "V2", "V3"))
    df_peaks <- subset(df_peaks, V1 %in% paste0("chr", c(1:22,"X")))
    
    print(nrow(df_peaks))
    
    peaks <- GRanges(seqnames = df_peaks$V1,
                     ranges = IRanges(start = df_peaks$V2,
                                      end = df_peaks$V3))
    
    peaks_overlap <- findOverlaps(query = peaks_all,
                                  subject = peaks,
                                  type = "any")
    peaks_overlap_idx <- peaks_overlap@from
    
    print(length(peaks_overlap_idx))
    
    df_true[[celltype]] <- 0
    df_true[[celltype]][peaks_overlap_idx] <- 1
}

write.table(df_true, file = "true_labels.txt", quote = FALSE, sep = "\t")
