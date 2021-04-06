import os
import subprocess

input_file = "../../../TagCount/TagCount.txt"
output_file = "../SAVER.txt"

job_name = "SAVER"
output_prefix = "SAVER"
subprocess.run(["sbatch", "-J", job_name,
                "-o", f"./cluster_out/{job_name}.txt",
                "-e", f"./cluster_err/{job_name}.txt",
                "--time", "120:00:00",
                "--mem", "180G",
                "-A", "rwth0233",
                "-c", "4",
                "run.zsh", input_file, output_file])
