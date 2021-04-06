import os
import subprocess

input_file = "../../../TagCount/TagCount.txt"
output_file = "../scBFA.txt"

job_name = "scBFA"
output_prefix = "scBFA"
subprocess.run(["sbatch", "-J", job_name,
                "-o", f"./cluster_out/{job_name}.txt",
                "-e", f"./cluster_err/{job_name}.txt",
                "--time", "30:00:00",
                "--mem", "180G",
                "-A", "rwth0233",
                "-c", "4",
                "run.zsh", input_file, output_file])
