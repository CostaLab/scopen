import os
import subprocess

method_list = ['MAGIC', 'scImpute', 'SAVER', 'cisTopic',
               'DCA', 'scBFA', 'SCALE', 'Raw', 'PCA',
               'scOpen']
method_list = ['SCALE']
num_clusters_list = [14, 15]

# create output directory
for num_clusters in num_clusters_list:
    output_dir = f"../k_{num_clusters}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

for num_clusters in num_clusters_list:
    for method in method_list:
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

        output_dir = f"../k_{num_clusters}/{method}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        job_name = f"k_{num_clusters}_{method}"
        subprocess.run(["sbatch", "-J", job_name,
                        "-o", f"./cluster_out/{job_name}.txt",
                        "-e", f"./cluster_err/{job_name}.txt",
                        "--time", "20:00:00",
                        "--mem", "180G",
                        "-A", "rwth0233",
                        "run.zsh", input_filename, output_dir, str(num_clusters)])
