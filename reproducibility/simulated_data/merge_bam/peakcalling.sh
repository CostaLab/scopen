#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J peakcalling
#SBATCH -e ./peakcalling.txt
#SBATCH -o ./peakcalling.txt
#SBATCH -t 30:00:00
#SBATCH --mem=60G

source ~/.zshrc
conda activate r-heart


macs2 callpeak -g hs --name ATAC --treatment ./B/ATAC.bam --outdir ./B --nomodel --nolambda --keep-dup all
macs2 callpeak -g hs --name ATAC --treatment ./CD4/ATAC.bam --outdir ./CD4 --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./CD8/ATAC.bam --outdir ./CD8 --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./CLP/ATAC.bam --outdir ./CLP --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./CMP/ATAC.bam --outdir ./CMP --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./ERY/ATAC.bam --outdir ./ERY --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./GMP/ATAC.bam --outdir ./GMP --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./HSC/ATAC.bam --outdir ./HSC --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./LMPP/ATAC.bam --outdir ./LMPP --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./MEP/ATAC.bam --outdir ./MEP --nomodel --nolambda --keep-dup all 
macs2 callpeak -g hs --name ATAC --treatment ./Mono/ATAC.bam --outdir ./Mono --nomodel --nolambda --keep-dup all  
macs2 callpeak -g hs --name ATAC --treatment ./MPP/ATAC.bam --outdir ./MPP --nomodel --nolambda --keep-dup all
macs2 callpeak -g hs --name ATAC --treatment ./NK/ATAC.bam --outdir ./NK --nomodel --nolambda --keep-dup all  

