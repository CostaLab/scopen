#!/bin/bash

### Job name
#SBATCH -J analysis
#SBATCH -e ./analysis.txt
#SBATCH -o ./analysis.txt

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH -t 120:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem=180G -c 12

source ~/.bashrc
conda activate r-4.0.3

Rscript compute.R

