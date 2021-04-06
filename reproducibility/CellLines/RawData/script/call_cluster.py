import os

input_loc = "./SRA"
cell_list = ['BJ', 'GM-rep1', 'GM-rep2', 'GM-rep3', 'GM-rep4', 'H1ESC', 'HL60', 'K562-rep1', 'K562-rep2', 'K562-rep3', 'K562-rep4', 'TF1']
cell_list = ['TF1']
indexes_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/data/Bowtie2_indexes/hg19/hg19"
for cell in cell_list:
    output_loc = os.path.join("/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/Nature2015/Runs", cell)
    os.system("mkdir -p " + output_loc)
    input_file = os.path.join(input_loc, "{}.txt".format(cell))
    f = open(input_file, "r")
    for line in f.readlines():
        sra = line.strip().split("\t")[0]
        job_name = "download_" + cell
        command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt "
        command += "-W 10:00 -M 10240 -S 100 -R \"select[hpcwork]\" -P rwth0233 ./run.zsh "
        os.system(command + sra + " " + output_loc + " " + indexes_loc)
