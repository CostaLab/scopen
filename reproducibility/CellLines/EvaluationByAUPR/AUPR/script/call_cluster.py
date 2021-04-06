import os
import subprocess

method_list = ['MAGIC', 'scImpute', 'SAVER', 'cisTopic',
               'DCA', 'scBFA', 'SCALE', 'Raw', 'PCA',
               'scOpen']

for method in method_list:
    if method == "Raw":
        input_filename = "../../../TagCount/TagCount.txt"
    elif method == "scImpute":
        input_filename = f"../../../Imputation/{method}/scimpute_count.txt"
    elif method == "SCALE":
        input_filename = f"../../../Imputation/{method}/imputed_data.txt"
    elif method == "DCA":
        input_filename = f"../../../Imputation/{method}/mean.tsv"
    else:
        input_filename = f"../../../Imputation/{method}/{method}.txt"

    output_filename = f"../{method}.txt"
    subprocess.run(["sbatch", "-J", method,
                    "-o", f"./cluster_out/{method}.txt",
                    "-e", f"./cluster_err/{method}.txt",
                    "--time", "3:00:00",
                    "--mem", "180G",
                    "-A", "rwth0233",
                    "run.zsh", input_filename, output_filename])
