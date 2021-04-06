#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate r-4.0.3

Rscript clustering.R -i $1 -o $2 -n $3 -m kmedoids
Rscript clustering.R -i $1 -o $2 -n $3 -m hc
