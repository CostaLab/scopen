import os

input_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/TagCount/TagCount.txt"
output_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Nature2015/Imputation/DCA"
job_name = "DCA"
command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + "./cluster_err/" + job_name + "_err.txt "
command += "-t 20:00:00 --mem 100G --partition=c18g --gres=gpu:1 ./run.zsh "
os.system(command + " " + input_file + " " + output_loc)
