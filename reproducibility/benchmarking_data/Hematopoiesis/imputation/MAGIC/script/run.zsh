#!/usr/bin/env zsh

################################################################
# PATH
export PATH=/home/rs619065/local/bin:$PATH
export PATH=/home/rs619065/.local/bin:$PATH
export PATH=/home/rs619065/local/bamtools/bin:$PATH
export PATH=/home/rs619065/local/miniconda2/bin:$PATH
export PATH=/usr/local_rwth/sw/cuda/9.0.176/bin:$PATH
export PATH=/home/rs619065/perl5/bin:$PATH
export PATH=/home/rs619065/meme/bin:$PATH
export PATH=/home/rs619065/homer/bin:$PATH
export PATH=/home/rs619065/opt/AMUSED:$PATH
export PATH=/home/rs619065/opt/IGVTools:$PATH


################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/rs619065/local/bamtools/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local_rwth/sw/cuda/9.0.176/lib64:$LD_LIBRARY_PATH

export R_LIBS_USER=$R_LIBS_USER:/home/rs619065/local/lib64/R/library
export PERL5LIB=/home/rs619065/perl5/lib/5.26.1:$PERL5LIB
export PERL5LIB=/home/rs619065/perl5/lib/perl5:$PERL5LIB

source activate py36env
Rscript magic.R --input=$1 --output=$2