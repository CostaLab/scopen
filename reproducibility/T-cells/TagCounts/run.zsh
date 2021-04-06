#!/bin/bash

### Job name
#SBATCH -J tagcount
#SBATCH -e ./tagcount.txt
#SBATCH -o ./tagcount.txt

### Time your job needs to execute, e. g. 15 min 30 sec
#SBATCH -t 20:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem=10G

################################################################
# PATH
export PATH=/home/rs619065/local/bin:$PATH
export PATH=/home/rs619065/.local/bin:$PATH

export PYTHONPATH=$PYTHONPATH:/home/rs619065/local/lib/python2.7/site-packages
export PYHTONPATH=$PYTHONPATH:/home/rs619065/.local/lib/python2.7/site-packages

################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/hpcwork/izkf/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH
export R_LIBS_USER=$R_LIBS_USER:/hpcwork/izkf/lib/R/library:/usr/bin/R/lib

input_dir=/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/NatureMedicine2018_V4/Merged
output_dir=/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/NatureMedicine2018_V4/TagCount

python get_tc.py $input_dir/ATACPeaks.merge.500.filter.bed $input_dir/ATAC.bam $output_dir/TagCount.txt
