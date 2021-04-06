import os

job_name = "merge"
command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
          "./cluster_err/" + job_name + "_err.txt "
command += "-W 48:00 -M 20480 -S 100 -R \"select[hpcwork]\" -P rwth0233 ./run.zsh "
os.system(command)