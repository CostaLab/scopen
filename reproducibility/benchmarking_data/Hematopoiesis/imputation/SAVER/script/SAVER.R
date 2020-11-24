library(optparse)
library(SAVER)

option_list = list(
  make_option("--input", type="character", default=NULL, help="input feature file"),
  make_option("--outdir", type="character", default=NULL, help="output file name")
)


opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

df <- read.table(opt$input)
df.saver <- saver(as.matrix(df), ncores = 12)
write.table(df.saver$estimate, file = opt$outdir, sep = "\t", quote = FALSE)
