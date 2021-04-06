#!/bin/bash
#
#### Job name
#SBATCH -J chromvar
#SBATCH -e ./chromvar.txt
#SBATCH -o ./chromvar.txt
#SBATCH -t 120:00:00
#SBATCH --mem=180G -c 8

source ~/.bashrc
conda activate r-4.0.3

time Rscript run_chromvar.R
