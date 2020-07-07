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


cd $2
samtools merge ATAC.bam $1/*.bam
samtools sort -o ATAC.sort.bam ATAC.bam --threads 32
mv ATAC.sort.bam ATAC.bam
samtools index ATAC.bam
igvtools count ATAC.bam ATAC.tdf hg19
