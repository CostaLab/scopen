# convert matrix file to the multiple column tab-separated vector
from __future__ import print_function
import sys

import pandas as pd

input_matrix_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_matrix_file, sep="\t", index_col=0)

print(df.index.values)

with open(output_file, "w") as output_f:
    for peak in df.index.values.tolist():
        for cell in df.columns.values.tolist():
            if df.at[peak, cell] > 0:
                output_f.write(peak + "\t" + cell + "\t" + str(df.at[peak, cell]) + "\n")
