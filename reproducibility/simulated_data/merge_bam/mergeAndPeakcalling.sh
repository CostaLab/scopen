samtools merge -f Merged.bam ./Merged/*/ATAC.bam -@ 100
samtools index Merged.bam
macs2 callpeak -g hs --name Merged --treatment Merged.bam --outdir ./ --nomodel --nolambda --keep-dup all 

