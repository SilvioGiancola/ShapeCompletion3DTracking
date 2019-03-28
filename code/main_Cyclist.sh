#!/bin/bash
#SBATCH --array=0-2
#SBATCH -J Cycl
#SBATCH -o Cycl.%3a.%A.out
#SBATCH -e Cycl.%3a.%A.err
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



echo "Define variables..."

if [ "$SLURM_ARRAY_TASK_ID" -eq "0" ]; then
   model_name="Cyclist_Ours";
   lambda_completion="1e-6"
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
   model_name="Cyclist_OnlyTracking";
   lambda_completion="0"
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
   model_name="Cyclist_OnlyCompletion";
   lambda_completion="1"
fi

echo "... Variables defined"


#run the training:
echo "Starting training python function ..."

python main.py --batch_size=64 --train_model \
--lambda_completion=$lambda_completion \
--model_name=$model_name \
--category_name=Cyclist \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/

echo "...training function Done"



#run the testing:
echo "Starting testing python function ..."

python main.py --test_model \
--lambda_completion=$lambda_completion \
--model_name=$model_name \
--category_name=Cyclist \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/

echo "...testing function Done"



