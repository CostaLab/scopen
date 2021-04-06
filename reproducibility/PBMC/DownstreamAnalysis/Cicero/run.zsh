#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J cicero
#SBATCH -e ./cicero.txt
#SBATCH -o ./cicero.txt
#SBATCH -t 120:00:00
#SBATCH --mem=180G -c 10 -A rwth0233

source ~/.zshrc
conda activate r-4.0.3

#python /home/rs619065/SingleCellOpenChromatin/MouseAltas/extract_exon_from_gtf.py \
#/home/rs619065/rgtdata/hg38/gencode.v24.annotation.gtf hg38.annotation.bed

Rscript run_cicero.R
