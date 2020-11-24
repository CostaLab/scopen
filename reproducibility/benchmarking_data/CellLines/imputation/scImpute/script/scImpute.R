library(optparse)
library(scImpute)

option_list = list(
  make_option("--input", type="character", default=NULL, help="input feature file"),
  make_option("--outdir", type="character", default=NULL, help="output file name")
)


opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

scimpute(count_path = opt$input, infile = "txt", outfile = "txt", 
         out_dir = opt$outdir, Kcluster = 6, ncores = 10)
