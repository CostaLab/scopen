#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J simulate
#SBATCH -e ./simulate.txt
#SBATCH -o ./simulate.txt
#SBATCH -t 120:00:00
#SBATCH --mem=180G -c 10 -A rwth0233

source ~/.zshrc
conda activate r-heart

Rscript -e rmarkdown::render"('simulate.Rmd',output_file='simulate.html',clean=TRUE)"
