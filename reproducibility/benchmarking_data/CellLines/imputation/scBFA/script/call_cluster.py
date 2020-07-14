import os

input_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/TagCount/TagCount.txt"
output_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/Imputation/scBFA/scBFA.txt"
job_name = "scBFA"
command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + "./cluster_err/" + job_name + "_err.txt "
command += "-t 120:00:00 --mem 100G -A rwth0233 ./run.zsh "
os.system(command + " " + input_file + " " + output_file)
