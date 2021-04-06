import os
import subprocess

method_list = ['MAGIC', 'scImpute', 'cisTopic',
               'scBFA', 'SCALE', 'Raw',
               'scOpen']

for method in method_list:
    job_name = method
    subprocess.run(["sbatch", "-J", job_name,
                    "-o", f"./cluster_out/{job_name}.txt",
                    "-e", f"./cluster_err/{job_name}.txt",
                    "--time", "10:00:00",
                    "--mem", "180G",
                    "-A", "rwth0233",
                    "run.zsh", method])
