#!/usr/bin/env zsh
#
#### Job name
#SBATCH -J peakcalling
#SBATCH -e ./peakcalling.txt
#SBATCH -o ./peakcalling.txt
#SBATCH -t 3:00:00
#SBATCH --mem=180G -c 48

source ~/.zshrc
conda activate r-4.0.3

for celltype in CD56_\(bright\)_NK_cells CD56_\(dim\)_NK_cells classical_monocytes effector_CD8_T_cells intermediate_monocytes MAIT_T_cells memory_B_cells memory_CD4_T_cells myeloid_DC naive_B_cells naive_CD4_T_cells naive_CD8_T_cells non-classical_monocytes plasmacytoid_DC
do
    macs2 callpeak -t ../../SplitedBam/BAM/PBMC_${celltype}.bam -n ${celltype} --outdir Peaks -g hs -f BAMPE --keep-dup all -q 0.01
done



