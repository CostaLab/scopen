import os

input_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Cell2018/TagCount/TagCount.txt"
output_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Cell2018/Imputation/MAGIC/MAGIC.txt"
job_name = "MAGIC"
command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
          "./cluster_err/" + job_name + "_err.txt "
command += "-t 10:00:00 --mem 30G ./run.zsh "
os.system(command + " " + input_file + " " + output_file)
