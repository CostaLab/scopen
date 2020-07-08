#!/usr/bin/env zsh

module load cuda/100
module load cudnn/7.4

dca $1 $2
