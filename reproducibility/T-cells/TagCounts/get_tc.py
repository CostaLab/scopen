import sys
import pysam
import numpy as np
import pandas as pd

from rgt.GenomicRegionSet import GenomicRegionSet

peak_file = sys.argv[1]
bam_file = sys.argv[2]
output_file = sys.argv[3]

forward_shift = 5
reverse_shift = -4

peaks = GenomicRegionSet("Peaks")
peaks.read(peak_file)
bam = pysam.AlignmentFile(bam_file, "rb")

# get sample name
sample_list = list()
f = open("/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/NatureMedicine2018_V4/Statistics/stat.txt")
f.readline()

for line in f.readlines():
    ll = line.strip().split("\t")
    sample_list.append(ll[0])    

sample_list.sort()

num_peaks = len(peaks)
num_samples = len(sample_list)
mat = np.zeros((num_peaks, num_samples), dtype=np.int16)

peak_name_list = list()
for i, peak in enumerate(peaks):
    peak_name_list.append("{}_{}_{}".format(peak.chrom, peak.initial, peak.final))
    for read in bam.fetch(peak.chrom, peak.initial, peak.final):
        if not read.is_reverse:
            cut_site = read.pos + forward_shift
        else:
            cut_site = read.aend + reverse_shift - 1
        
        if peak.initial <= cut_site <= peak.final:
            sample = read.query_name.split(".")[0]
            j = sample_list.index(sample)
            mat[i, j] += 1

df = pd.DataFrame(mat, columns=sample_list, index=peak_name_list)

df.to_csv(output_file, sep="\t")
