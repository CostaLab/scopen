library(optparse)
library(Matrix)
library(Rmagic)
library(methods)

option_list = list(
  make_option("--input", type="character", default=NULL, help="input feature file"),
  make_option("--output", type="character", default=NULL, help="output file name")
)


opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

df <- read.table(opt$input, header = TRUE)
df.t <- t(df)
df.t <- library.size.normalize(df.t)
df.t <- sqrt(df.t)
magic_out <- magic(data = df.t, seed = 42)

write.table(t(magic_out$result), file = opt$output, quote = FALSE, sep = "\t")
