#!/usr/bin/env zsh

################################################################
# PATH
export PATH=/usr/bin:$PATH
export PATH=/usr/local_host/bin:$PATH
export PATH=/home/rs619065/local/bin:$PATH
export PATH=/home/rs619065/.local/bin:$PATH
export PATH=/home/rs619065/local/bamtools/bin:$PATH
export PATH=/usr/local_rwth/sw/cuda/9.0.176/bin:$PATH
export PATH=/home/rs619065/perl5/bin:$PATH
export PATH=/home/rs619065/meme/bin:$PATH
export PATH=/home/rs619065/homer/bin:$PATH
export PATH=/home/rs619065/opt/AMUSED:$PATH
export PATH=/home/rs619065/opt/IGVTools:$PATH

export PYTHONPATH=/home/rs619065/local/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=/home/rs619065/.local/lib/python2.7/site-packages:$PYTHONPATH

################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/rs619065/local/bamtools/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local_rwth/sw/cuda/9.0.176/lib64:$LD_LIBRARY_PATH

export R_LIBS_USER=$R_LIBS_USER:/home/rs619065/local/lib64/R/library
export PERL5LIB=/home/rs619065/perl5/lib/5.26.1:$PERL5LIB
export PERL5LIB=/home/rs619065/perl5/lib/perl5:$PERL5LIB

export RUBYLIB=$RUBYLIB:/home/rs619065/AMUSED:/home/rs619065/Ruby-DNA-Tools


#cd /hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/Nature2015/Merged
#samtools merge ATAC.bam /hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/Nature2015/Runs/*/*.bam
#samtools sort -o ATAC.sort.bam ATAC.bam --threads 32
#mv ATAC.sort.bam ATAC.bam
#samtools index ATAC.bam
#mkdir Peaks
#macs2 callpeak -t ATAC.bam -n ATAC --outdir Peaks -g hs -f BAM --keep-dup all --call-summits

awk -v OFS='\t' '{print $1,$2-250,$3+249,$4,$5}' ../Peaks/ATAC_summits.bed > ../ATACPeaks.bed
bedtools merge -i ../ATACPeaks.bed -c 5 -o max > ../ATACPeaks.merge.bed
awk '{printf "%s\t%i\t%i\t%s\n", $1,($2+$3)/2-250,($2+$3)/2+250,$4}' ../ATACPeaks.merge.bed > ../ATACPeaks.merge.500.bed

blacklist_loc=/hpcwork/izkf/projects/SingleCellOpenChromatin/data/blacklists/hg19-human
bedtools intersect -wa -a ../ATACPeaks.merge.500.bed \
-b $blacklist_loc/Anshul_Hg19UltraHighSignalArtifactRegions.bed $blacklist_loc/Duke_Hg19SignalRepeatArtifactRegions.bed \
$blacklist_loc/wgEncodeHg19ConsensusSignalArtifactRegions.bed -v > ../ATACPeaks.merge.500.filter.bed
