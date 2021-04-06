#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J snapATAC
#SBATCH -e ./snapATAC.txt
#SBATCH -o ./snapATAC.txt
#SBATCH -t 5:00:00
#SBATCH --mem=180G -c 24 -A rwth0233

source ~/.zshrc
conda activate r-4.0.3


Rscript createFragment.R

snaptools snap-pre \
--input-file=fragment.bed \
--output-snap=fragment.snap \
--genome-name=hg19 \
--genome-size=/home/rs619065/rgtdata/hg19/chrom.sizes.hg19 \
--min-flen=0 \
--max-flen=2000 \
--keep-chrm=TRUE \
--keep-single=False \
--keep-secondary=False \
--overwrite=True \
--max-num=1000000 \
--min-cov=100 \
--verbose=True

snaptools snap-add-bmat \
--snap-file=fragment.snap \
--bin-size-list 5000 \
--verbose=True


Rscript snapATAC.R

