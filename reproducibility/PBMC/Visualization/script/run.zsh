#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate r-4.0.3

Rscript visualize_UMAP.R --input $1 --output_dir ../UMAP --output_filename $2
