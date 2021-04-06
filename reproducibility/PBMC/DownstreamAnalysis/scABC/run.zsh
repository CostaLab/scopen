#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J scABC
#SBATCH -e ./scABC.txt
#SBATCH -o ./scABC.txt
#SBATCH -t 120:00:00
#SBATCH --mem=180G -c 10 -A rwth0233

source ~/.zshrc
conda activate r-4.0.3

Rscript scABC.R
