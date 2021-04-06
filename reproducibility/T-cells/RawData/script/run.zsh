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
export PATH=/home/rs619065/miniconda2/bin:$PATH


################################################################
# LIBRARYPATH
export LD_LIBRARY_PATH=/home/rs619065/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/rs619065/local/bamtools/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local_rwth/sw/cuda/9.0.176/lib64:$LD_LIBRARY_PATH

export R_LIBS_USER=$R_LIBS_USER:/home/rs619065/local/lib64/R/library
export PERL5LIB=/home/rs619065/perl5/lib/5.26.1:$PERL5LIB
export PERL5LIB=/home/rs619065/perl5/lib/perl5:$PERL5LIB

export RUBYLIB=$RUBYLIB:/home/rs619065/AMUSED:/home/rs619065/Ruby-DNA-Tools

# This script is designed to process single cell ATAC-seq data using C1 system with paired-end sequencing
# The input is sequence read archive (SRA) number.

# downloading of SRA
prefetch -v $1 &&
cd $2 &&

# Dump each read into separate file. Files will receive suffix corresponding to read number.
fastq-dump --split-3 /hpcwork/izkf/ncbi/sra/$1.sra &&
rm /hpcwork/izkf/ncbi/sra/$1.sra &&

# Adapter sequences were trimmed from FASTQs
trim_galore --suppress_warn -q 30 --paired -o ./ $1_1.fastq $1_2.fastq
rm $1_1.fastq &&
rm $1_2.fastq &&

# Map the reads to reference genome
bowtie2 --very-sensitive --no-discordant -x $3 -1 $1_1_val_1.fq -2 $1_2_val_2.fq -S $1.map.sam -X 2000 -p 12 &&
rm $1_1_val_1.fq &&
rm $1_2_val_2.fq &&

# Filter out reads mapped to chrY, mitochondria, and unassembled "random" contigs, 
sed -i '/chrY/d;/chrM/d;/random/d;/chrUn/d' $1.map.sam &&

# Convert sam file to bam file, sort the result and generate the index file
samtools view -Sb $1.map.sam > $1.map.bam &&
samtools sort $1.map.bam -o $1.sort.bam &&
samtools index $1.sort.bam &&
rm $1.map.sam &&
rm $1.map.bam &&

# Remove duplicates 
java -jar /hpcwork/izkf/jar/picard.jar MarkDuplicates INPUT=$1.sort.bam \
OUTPUT=$1.rmdup.bam METRICS_FILE=$1_matrics.txt REMOVE_DUPLICATES=true VALIDATION_STRINGENCY=LENIENT &&
rm $1.sort.bam
rm $1.sort.bam.bai

# and bad map quality
samtools view -bq 30 $1.rmdup.bam > $1.filter.bam &&
samtools index $1.filter.bam &&
rm $1.rmdup.bam

# Require reads to be properly paired
samtools view -f2 $1.filter.bam -b > $1.bam &&
samtools index $1.bam &&
samtools flagstat $1.bam > $1_qc.txt &&

rm $1.filter.bam
rm $1.filter.bam.bai
