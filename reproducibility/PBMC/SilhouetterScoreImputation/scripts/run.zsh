#!/usr/bin/env zsh

source ~/.zshrc
conda activate r-4.0.3

Rscript compute.R -m $1
