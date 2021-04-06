#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J peakcalling
#SBATCH -e ./peakcalling.txt
#SBATCH -o ./peakcalling.txt
#SBATCH -t 3:00:00
#SBATCH --mem=180G -c 48

source ~/.zshrc
conda activate r-heart

input_loc=/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/Runs


#mkdir -p BJ
#mkdir -p GM12878
#mkdir -p H1ESC
#mkdir -p HL60
#mkdir -p K562
#mkdir -p TF1

#samtools merge ./BJ/ATAC.bam ${input_loc}/BJ/*.bam --threads 48
#samtools merge ./GM12878/ATAC.bam ${input_loc}/GM-rep*/*.bam --threads 48
#samtools merge ./H1ESC/ATAC.bam ${input_loc}/H1ESC/*.bam --threads 48
#samtools merge ./HL60/ATAC.bam ${input_loc}/HL60/*.bam --threads 48
#samtools merge ./K562/ATAC.bam ${input_loc}/K562-rep*/*.bam --threads 48
#samtools merge ./TF1/ATAC.bam ${input_loc}/TF1/*.bam --threads 48


#samtools index ./BJ/ATAC.bam
#samtools index ./GM12878/ATAC.bam
#samtools index ./H1ESC/ATAC.bam
#samtools index ./HL60/ATAC.bam
#samtools index ./K562/ATAC.bam
#samtools index ./TF1/ATAC.bam


for celltype in BJ GM12878 H1ESC HL60 K562 TF1
do
    macs2 callpeak -t ./${celltype}/ATAC.bam -n ${celltype} --outdir Peaks -g hs -f BAMPE --keep-dup all -q 0.01
done



