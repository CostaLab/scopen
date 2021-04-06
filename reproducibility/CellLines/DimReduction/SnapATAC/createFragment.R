suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(chromstaR))
suppressPackageStartupMessages(library(doParallel))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(BSgenome.Hsapiens.UCSC.hg19))
suppressPackageStartupMessages(library(raster))


df <- read.table("../TagCount/TagCount.txt", header = TRUE,
                 row.names = 1, sep = "\t")

df_anno <- read.table("../Statistics/stat.txt", header = TRUE) %>%
    subset(., Runs %in% colnames(df))

fragments_list <- vector(mode = "list", length = nrow(df))
idx <- 1

for(i in 1:nrow(df_anno)) {
    bamfile <- sprintf("../Runs/%s/%s.bam", df_anno[i, "CellType"], 
                       df_anno[i, "Runs"])
    fragments_list[[idx]] <- readBamFileAsGRanges(bamfile = bamfile,
                                                  pairedEndReads = TRUE,
                                                  max.fragment.width = 2000)
    fragments_list[[idx]]$cell <- df_anno[i, "Runs"]
    idx <- idx + 1
}
fragments <- do.call(c, fragments_list)
df <- as.data.frame(fragments) %>%
    subset(., select = c("seqnames", "start", "end", "cell"))

write.table(df, file = "fragment.bed", col.names = FALSE, sep = "\t",
            quote = FALSE, row.names = FALSE)
