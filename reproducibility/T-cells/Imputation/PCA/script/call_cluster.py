import os
import subprocess

input_file = "../../../TagCount/TagCount.txt"
output_file = "../PCA.txt"

job_name = "PCA"
output_prefix = "PCA"
subprocess.run(["sbatch", "-J", job_name,
                "-o", f"./cluster_out/{job_name}.txt",
                "-e", f"./cluster_err/{job_name}.txt",
                "--time", "12:00:00",
                "--mem", "180G",
                "-A", "rwth0233",
                "-c", "4",
                "run.zsh", input_file, output_file])
