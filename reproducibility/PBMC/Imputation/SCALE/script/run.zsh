#!/usr/local_rwth/bin/zsh

module load cuda

source ~/.zshrc
conda activate r-4.0.3

start=$(date +'%s')

time python ~/SCALE/SCALE.py -d $1 -o $2 --impute --seed 42 --lr 0.0002 --min_peaks 0 -x 0 --high 1

echo "It took $(($(date +'%s') - $start)) seconds"
