import os

input_loc = "./SRA"
cell_list = ['CLP', 'CMP', 'GMP', 'HSC', 'LMPP', 'MEP', 'MPP', 'pDC']
indexes_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/data/Bowtie2_indexes/hg19/hg19"
for cell in cell_list:
    output_loc = os.path.join("/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/Cell2018/Runs", cell)
    os.system("mkdir -p " + output_loc)
    input_file = os.path.join(input_loc, "{}.txt".format(cell))
    f = open(input_file, "r")
    for line in f.readlines():
        sra = line.strip().split("\t")[0]
        bam_file = os.path.join(output_loc, "{}.bam".format(sra))
        if os.path.isfile(bam_file):
            continue
        job_name = "download_" + cell
        command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt "
        command += "-W 10:00 -M 20480 -S 100 -R \"select[hpcwork]\" -P rwth0233 ./run.zsh "
        os.system(command + sra + " " + output_loc + " " + indexes_loc)
