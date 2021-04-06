#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J compute
#SBATCH -e ./compute.txt
#SBATCH -o ./compute.txt
#SBATCH -t 10:00:00
#SBATCH --mem=180G -c 24 -A rwth0233

source ~/.zshrc
conda activate r-4.0.3

Rscript compute.R
