#!/usr/local_rwth/bin/zsh

module load cuda/100
module load cudnn/7.4

source ~/.zshrc
conda activate r-4.0.3

start=$(date +'%s')

time dca $1 $2

echo "It took $(($(date +'%s') - $start)) seconds"
