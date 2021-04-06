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

celltype_list <- c("Jurkat_T_cell", "Memory_T_cell", "Naive_T_cell", "Th17_T_cell")
df_true <- data.frame(matrix(ncol = length(celltype_list), nrow = nrow(df)))
colnames(df_true) <- celltype_list
rownames(df_true) <- rownames(df)

for(celltype in celltype_list){
    df_peaks <- read.table(sprintf("../CellTypePeaks/Peaks/%s_peaks.narrowPeak",
                                   celltype))
    df_peaks <- subset(df_peaks, select = c("V1", "V2", "V3"))
    peaks <- GRanges(seqnames = df_peaks$V1,
                     ranges = IRanges(start = df_peaks$V2,
                                      end = df_peaks$V3))

    peaks_overlap <- findOverlaps(query = peaks_all,
                                  subject = peaks,
                                  type = "any")
    peaks_overlap_idx <- peaks_overlap@from
    df_true[[celltype]] <- 0
    df_true[[celltype]][peaks_overlap_idx] <- 1
}

write.table(df_true, file = "true_labels.txt", quote = FALSE, sep = "\t")
