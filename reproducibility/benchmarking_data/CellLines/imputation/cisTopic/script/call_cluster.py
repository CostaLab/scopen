import os

input_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/TagCount/TagCount.txt"
output_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/Imputation/cisTopic/cisTopic.txt"
job_name = "cisTopic"
command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + "./cluster_err/" + job_name + "_err.txt "
command += "-t 10:00:00 --mem 180G -c 48 -A rwth0233 ./run.zsh "
os.system(command + " " + input_file + " " + output_file)
