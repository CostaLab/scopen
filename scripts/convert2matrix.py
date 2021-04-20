# convert the multiple column tab-separated vector to a matrix
import sys

import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]


peaks = []
cells = []
with open(input_file) as f:
    for line in f:
        ll = line.strip().split("\t")
        peaks.append(ll[0])
        cells.append(ll[1])

peaks = list(set(peaks))
cells = list(set(cells))

n_peaks = len(peaks)
n_cells = len(cells)

df = pd.DataFrame(0, columns=cells, index=peaks)

with open(input_file) as f:
    for line in f:
        ll = line.strip().split("\t")
        peak = ll[0]
        cell = ll[1]
        df.at[peak, cell] = int(ll[2])

df.to_csv(output_file, sep="\t")