import os
import subprocess

method_list = ['MAGIC', 'scImpute', 'SAVER', 'cisTopic',
               'DCA', 'scBFA', 'SCALE', 'Raw', 'PCA',
               'scOpen']


for method in method_list:
    job_name = method
    if method == "Raw":
        input_filename = "../../TagCount/TagCount.txt"
    elif method == "scImpute":
        input_filename = f"../../Imputation/{method}/scimpute_count.txt"
    elif method == "SCALE":
        input_filename = f"../../Imputation/{method}/imputed_data.txt"
    elif method == "DCA":
        input_filename = f"../../Imputation/{method}/mean.tsv"
    else:
        input_filename = f"../../Imputation/{method}/{method}.txt"

    subprocess.run(["sbatch", "-J", job_name,
                    "-o", f"./cluster_out/{job_name}.txt",
                    "-e", f"./cluster_err/{job_name}.txt",
                    "--time", "120:00:00",
                    "-A", "rwth0233",
                    "--mem", "180G",
                    "run.zsh", input_filename, method])
