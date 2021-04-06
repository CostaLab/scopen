#!/usr/local_rwth/bin/zsh

source ~/.zshrc
conda activate r-4.0.3

start=$(date +'%s')

time scopen --input $1 --input_format 10X --output_dir $2 --output_prefix $3 --output_format dense --verbose 0 --nc 1 --estimate_rank --no_impute --min_n_components 10 --max_n_components 40 --step_n_components 2

echo "It took $(($(date +'%s') - $start)) seconds"
