library(optparse)
library(cisTopic)
library(methods)
library(stringr)

option_list = list(
  make_option("--input", type="character", default=NULL, help="input feature file"),
  make_option("--output", type="character", default=NULL, help="output file name")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

input_file <- opt$input
x <- read.table(input_file, header = TRUE)
region <- as.data.frame(t(as.data.frame(str_split(rownames(x), "_"))))
colnames(region) <- c("chrom", "p1", "p2")
rownames(x) <- paste0(region$chrom, ":", region$p1, "-", region$p2)

cisTopicObject <- createcisTopicObject(count.matrix = as.matrix(x))

cisTopicObject <- runModels(cisTopicObject, 
                            topic = c(5:50), 
                            seed = 987, nCores = 48, addModels=FALSE)

cisTopicObject <- selectModel(cisTopicObject)

t1 <- as.data.frame(cisTopicObject@selected.model$document_expects)
t2 <- as.data.frame(t(cisTopicObject@selected.model$topics))

#colnames(t1) <- colnames(x)
#write.table(t1, file = opt$output, quote = FALSE, sep = "\t")

# normalization
t1 <- t1 / rowSums(t1)
t2 <- t2 / colSums(t2)

x_complete <- as.matrix(t2) %*% as.matrix(t1)

colnames(x_complete) <- colnames(x)
rownames(x_complete) <- rownames(x)

write.table(x_complete, file = opt$output, quote = FALSE, sep = "\t")
