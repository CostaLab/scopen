import os
import sys

cell_list = ['HSC', 'MPP', 'LMPP', 'CMP', 'GMP', 'MEP', 'Mono', 'CD4', 'CD8', 'NK', 'B', 'CLP', 'ERY']
cell_sra_dict = dict()

cell_sra_dict['HSC'] = ['SRR2920477', 'SRR2920478', 'SRR2920531', 'SRR2920532', 'SRR2920466', 'SRR2920506', 'SRR2920507']
cell_sra_dict['MPP'] = ['SRR2920479', 'SRR2920533', 'SRR2920534', 'SRR2920467', 'SRR2920509', 'SRR2920510']
cell_sra_dict['LMPP'] = ['SRR2920480', 'SRR2920535', 'SRR2920468']
cell_sra_dict['CMP'] = ['SRR2920481', 'SRR2920482', 'SRR2920536', 'SRR2920537', 'SRR2920469', 'SRR2920470', 'SRR2920500', 'SRR2920501']
cell_sra_dict['GMP'] = ['SRR2920483', 'SRR2920484', 'SRR2920538', 'SRR2920539', 'SRR2920471', 'SRR2920472', 'SRR2920505']
cell_sra_dict['MEP'] = ['SRR2920485', 'SRR2920486', 'SRR2920540', 'SRR2920541', 'SRR2920473', 'SRR2920474', 'SRR2920508']
cell_sra_dict['Mono'] = ['SRR2920487', 'SRR2920488', 'SRR2920542', 'SRR2920543', 'SRR2920475', 'SRR2920476']
cell_sra_dict['CD4'] = ['SRR2920493', 'SRR2920514', 'SRR2920496', 'SRR2920518', 'SRR2920519']
cell_sra_dict['CD8'] = ['SRR2920494', 'SRR2920515', 'SRR2920497', 'SRR2920520', 'SRR2920521']
cell_sra_dict['NK'] = ['SRR2920495', 'SRR2920516', 'SRR2920511', 'SRR2920512', 'SRR2920526', 'SRR2920527']
cell_sra_dict['B'] = ['SRR2920492', 'SRR2920513', 'SRR2920517', 'SRR2920544']
cell_sra_dict['CLP'] = ['SRR2920498', 'SRR2920499', 'SRR2920522', 'SRR2920528', 'SRR2920545']
cell_sra_dict['ERY'] = ['SRR2920502', 'SRR2920503', 'SRR2920504', 'SRR2920523', 'SRR2920524', 'SRR2920525', 'SRR2920529', 'SRR2920530']

index_loc = "/hpcwork/izkf/projects/SingleCellOpenChromatin/data/Bowtie2_indexes/hg19/hg19"
for cell in cell_list:
    output_loc = os.path.join("/hpcwork/izkf/projects/SingleCellOpenChromatin/data/ATAC/NatureGenetics2016/Runs", cell)
    os.system("mkdir -p " + output_loc)
    for sra in cell_sra_dict[cell]:
        job_name = sra
        command = "bsub -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt "
        command += "-W 120:00 -M 51200 -S 100 -R \"select[hpcwork]\" -P rwth0233 ./run.zsh "
        os.system(command + sra + " " + output_loc + " " + index_loc)
