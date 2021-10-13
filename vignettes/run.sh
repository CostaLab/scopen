#!/bin/bash

### Job name
#SBATCH -J PBMC
#SBATCH -e ./PBMC.txt
#SBATCH -o ./PBMC.txt

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH -t 120:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem=100G -c 96

source ~/.bashrc
conda activate r-4.0.3

Rscript -e rmarkdown::render"('signac_pbmc.Rmd',output_file='signac_pbmc.html',clean=TRUE)"
jupyter nbconvert --to html --execute ./epiScanpy.ipynb --output-dir ./
