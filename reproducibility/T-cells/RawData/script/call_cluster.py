import os
import pandas as pd

input_file = "./SraRunTable.txt"
indexes_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/local/Bowtie2_indexes/hg19/hg19"


df = pd.read_csv(input_file, sep = "\t")
df.cell_type.replace(['CD4+ CD26- T cell', 'CD4+ CD26+ T cell', 'CD4+ T cell', 'Memory T cell', 'Naive T cell', 'single Jurkat T cell', 'Th17 T cell'],
                     ['CD4_CD26_minus_T_cell', 'CD4_CD26_plus_T_cell', 'CD4_T_cell', 'Memory_T_cell', 'Naive_T_cell', 'Jurkat_T_cell', 'Th17_T_cell'], inplace=True)


sras = df.Run.values
cell_types = df.cell_type.values

for i, cell_type in enumerate(cell_types):
    sra = sras[i]
    output_loc = os.path.join("/hpcwork/izkf/projects/SingleCellOpenChromatin/local/ATAC/NatureMedicine2018/Runs", cell_type)
    os.system("mkdir -p " + output_loc)
    job_name = sra
    command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + "./cluster_err/" + job_name + "_err.txt "
    command += "-t 30:00:00 --mem 100G -c 12 -A rwth0233 ./run.zsh "
    os.system(command + sra + " " + output_loc + " " + indexes_loc)
