module load cuda
python ~/SCALE/SCALE.py -d $1 -o $2 --impute --seed 42 --lr 0.0002 --min_peaks 0 -x 0.02
