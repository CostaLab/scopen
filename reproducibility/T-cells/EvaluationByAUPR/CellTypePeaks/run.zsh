#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J peakcalling
#SBATCH -e ./peakcalling.txt
#SBATCH -o ./peakcalling.txt
#SBATCH -t 3:00:00
#SBATCH --mem=180G -c 48

source ~/.zshrc
conda activate r-4.0.3

input_loc=/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/NatureMedicine2018/Runs


#mkdir -p Jurkat_T_cell
#mkdir -p Memory_T_cell
#mkdir -p Naive_T_cell
#mkdir -p Th17_T_cell

#samtools merge ./Jurkat_T_cell/ATAC.bam ${input_loc}/Jurkat_T_cell/*.bam --threads 48
#samtools merge ./Memory_T_cell/ATAC.bam ${input_loc}/Memory_T_cell/*.bam --threads 48
#samtools merge ./Naive_T_cell/ATAC.bam ${input_loc}/Naive_T_cell/*.bam --threads 48
#samtools merge ./Th17_T_cell/ATAC.bam ${input_loc}/Th17_T_cell/*.bam --threads 48


#samtools index ./Jurkat_T_cell/ATAC.bam
#samtools index ./Memory_T_cell/ATAC.bam
#samtools index ./Naive_T_cell/ATAC.bam
#samtools index ./Th17_T_cell/ATAC.bam


for celltype in Jurkat_T_cell Memory_T_cell Naive_T_cell Th17_T_cell
do
    macs2 callpeak -t ./${celltype}/ATAC.bam -n ${celltype} --outdir Peaks -g hs -f BAMPE --keep-dup all -q 0.01
done


