import os

input_file = "/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/Nature2015/TagCount/TagCount.txt"
output_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/Nature2015/Imputation/scImpute"
job_name = "scImpute"
command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
            "./cluster_err/" + job_name + "_err.txt "
command += "-W 120:00 -M 51200 -S 100 -R \"select[hpcwork]\" -P rwth0233 ./run.zsh "
os.system(command + " " + input_file + " " + output_loc)
