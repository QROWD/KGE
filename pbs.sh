#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load gcc
module load python
module load cuda-toolkit/8.0.44
module load cudnn/5.1

export PATH="/mnt/nfs/home/lpfgarcia/conda/bin:$PATH"
export DEVICE=cuda0

cd /mnt/nfs/home/lpfgarcia/KGE/
python run.py evaluation --model $model --data $data --k $k --epoch $epoch --folds $folds > $model.$data.$k.$epoch.$folds.log

