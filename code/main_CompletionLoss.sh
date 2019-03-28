#!/bin/bash
#SBATCH --array=1-8
#SBATCH -J Comp
#SBATCH -o Comp.%3a.%A.out
#SBATCH -e Comp.%3a.%A.err
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
--model_name=Comp1e-$SLURM_ARRAY_TASK_ID \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/ \
--lambda_completion=1e-$SLURM_ARRAY_TASK_ID

echo "...training function Done"



#run the testing:
echo "Starting testing python function ..."

python main.py --test_model \
--model_name=Comp1e-$SLURM_ARRAY_TASK_ID \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/

echo "...testing function Done"



