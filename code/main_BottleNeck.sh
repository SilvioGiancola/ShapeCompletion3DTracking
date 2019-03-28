#!/bin/bash
#SBATCH --array=[8,16,32,64,128,256,512,1024]
#SBATCH -J BNeck
#SBATCH -o BNeck.%3a.%A.out
#SBATCH -e BNeck.%3a.%A.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=9
#SBATCH --mem=40G
#SBATCH --reservation=IVUL

# activate your conda env
echo "Loading anaconda..."

module purge
module load anaconda3

conda create -y -n ShapeCompletion3DTracking python tqdm numpy pandas shapely matplotlib pomegranate
source activate ShapeCompletion3DTracking
conda install -y pytorch=0.4.1 cuda90 -c pytorch
pip install pyquaternion 

echo "...Anaconda env loaded"



#run the training:
echo "Starting training python function ..."

python main.py --batch_size=64 --train_model \
--model_name=BottleNeck$SLURM_ARRAY_TASK_ID \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/ \
--bneck_size=$SLURM_ARRAY_TASK_ID

echo "...training function Done"



#run the testing:
echo "Starting testing python function ..."

python main.py --test_model \
--model_name=BottleNeck$SLURM_ARRAY_TASK_ID \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/ \
--bneck_size=$SLURM_ARRAY_TASK_ID

echo "...testing function Done"



