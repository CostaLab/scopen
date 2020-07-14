library(optparse)
library(scBFA)
library(methods)
library(stringr)
library(SingleCellExperiment)

option_list = list(
  make_option("--input", type="character", default=NULL, help="input feature file"),
  make_option("--output", type="character", default=NULL, help="output file name")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

input_file <- opt$input
x <- read.table(input_file, header = TRUE)
bfa_model <- scBFA(scData = as.matrix(x), numFactors = 30, method = "CG")

zz <- as.matrix(bfa_model$ZZ)
aa <- as.matrix(bfa_model$AA)

x_complete <- aa %*% t(zz)

colnames(x_complete) <- colnames(x)
rownames(x_complete) <- rownames(x)

write.table(x_complete, file = opt$output, quote = FALSE, sep = "\t")
