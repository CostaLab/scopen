import os

cell_list = ['HSC', 'MPP', 'LMPP', 'CMP', 'GMP', 'MEP', 'Mono', 'CD4', 'CD8', 'NK', 'B', 'CLP', 'ERY']
for cell in cell_list:
    input_dir = os.path.join("/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/NatureGenetics2016/Runs", cell)
    output_dir = os.path.join("/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/NatureGenetics2016/Merged", cell)
    os.system("mkdir -p " + output_dir)
    job_name = "merge"
    command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
              "./cluster_err/" + job_name + "_err.txt "
    command += "-W 48:00 -M 51200 -S 100 -R \"select[hpcwork]\" -P rwth0233 ./run.zsh "
    os.system(command + " " + input_dir + " " + output_dir)
