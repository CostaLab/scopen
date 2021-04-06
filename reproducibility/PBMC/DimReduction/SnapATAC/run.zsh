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

#input_dir=/hpcwork/izkf/projects/Multiome/local/ATAC_GEX/PBMC_from_a_healthy_donor_granulocytes_removed_through_cell_sorting_10k/output

#gunzip ${input_dir}/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz 
#sort -k4,4 ${input_dir}/pbmc_granulocyte_sorted_10k_atac_fragments.tsv > fragments.bed
#gzip fragments.bed

#sed -e "s/-1//g" -i fragments.bed

snaptools snap-pre \
--input-file=fragments.bed \
--output-snap=fragment.snap \
--barcode-file=./barcodes.txt \
--genome-name=hg38 \
--genome-size=/home/rs619065/rgtdata/hg38/chrom.sizes.hg38 \
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

