#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J snapATAC
#SBATCH -e ./snapATAC.txt
#SBATCH -o ./snapATAC.txt
#SBATCH -t 120:00:00
#SBATCH --mem=180G -c 24 -A rwth0233

source ~/.zshrc
conda activate r-4.0.3


#for sample in D0_1 D0_2 D2_1 D2_2 D10_1 D10_2
#do
#gunzip -c ../../CountMatrix3/${sample}/outs/fragments.tsv.gz | sort -k4,4 > ${sample}.bed &&
#gzip ${sample}.bed
#snaptools snap-pre \
#--input-file=./${sample}.bed.gz \
#--output-snap=./${sample}.snap \
#--genome-name=hg38 \
#--genome-size=/home/rs619065/rgtdata/hg38/chrom.sizes.hg38 \
#--min-flen=0 \
#--max-flen=2000 \
#--keep-chrm=TRUE \
#--keep-single=False \
#--keep-secondary=False \
#--overwrite=True \
#--max-num=1000000 \
#--min-cov=100 \
#--verbose=True

#snaptools snap-add-bmat \
#--snap-file=./${sample}.snap \
#--bin-size-list 5000

#done

Rscript snapATAC.R

