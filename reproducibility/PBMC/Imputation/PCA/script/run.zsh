#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate r-4.0.3

start=$(date +'%s')

Rscript /hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/scripts/Imputation/pca.R --input=$1 --output=$2

echo "It took $(($(date +'%s') - $start)) seconds"
