#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate r-4.0.3

start=$(date +'%s')

time scopen --input $1 --input_format dense --output_dir $2 --output_prefix $3 --output_format dense --verbose 0 --estimate_rank --nc 4

echo "It took $(($(date +'%s') - $start)) seconds"
