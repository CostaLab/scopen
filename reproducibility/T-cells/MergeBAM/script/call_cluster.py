import os

job_name = "merge"
command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
          "./cluster_err/" + job_name + "_err.txt -t 100:00:00 --mem 100G -c 24 -A rwth0233 "
command += "./run.zsh "
os.system(command)
