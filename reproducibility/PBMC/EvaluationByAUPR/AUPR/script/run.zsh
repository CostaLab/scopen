#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate r-4.0.3

Rscript compute_aupr.R -i $1 -o $2
