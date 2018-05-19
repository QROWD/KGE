#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load gcc
module load python
module load cuda/tools
module load cudnn/5.1

export PATH="/mnt/nfs/home/lpfgarcia/conda/bin:$PATH"

cd /mnt/nfs/home/lpfgarcia/KGE/
DEVICE=cuda0 python run.py evaluation --model $model --data $data --k $k --epoch $epoch --folds $folds > $model.$data.$k.$epoch.$folds.log
