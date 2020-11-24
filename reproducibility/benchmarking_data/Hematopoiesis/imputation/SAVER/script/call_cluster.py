import os

input_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Cell2018/TagCount/TagCount.txt"
output_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/Cell2018/Imputation/SAVER/SAVER.txt"
job_name = "saver"
command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
          "./cluster_err/" + job_name + "_err.txt "
command += "-t 120:00:00 --mem 150G -c 24 -N 2 ./run.zsh "
os.system(command + " " + input_file + " " + output_loc)
