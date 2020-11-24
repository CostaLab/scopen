library(optparse)
library(SAVER)
library(doParallel)

option_list = list(
  make_option("--input", type="character", default=NULL, help="input feature file"),
  make_option("--outdir", type="character", default=NULL, help="output file name")
)


opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

df <- read.table(opt$input)

cl <- makeCluster(10, outfile = "")
registerDoParallel(cl)
out_df <- saver(as.matrix(df))
stopCluster(cl)

res <- out_df$estimate
write.table(res, file = opt$outdir, sep = "\t", quote = FALSE)
