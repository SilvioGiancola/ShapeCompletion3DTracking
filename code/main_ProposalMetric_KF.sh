#!/bin/bash
#SBATCH --array=[1,10,20,40,60,80,100,120,140,160]
#SBATCH -J Prop_KF
#SBATCH -o %x.%3a.%A.out
#SBATCH -e %x.%3a.%A.err
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


# if [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
#    search_space="Particle";
#    reference_BB="previous_result"
# fi

# if [ "$SLURM_ARRAY_TASK_ID" -eq "3" ]; then
#    search_space="GMM25";
#    reference_BB="previous_result"
# fi

echo "... Varaibles defined"



#run the testing:
echo "Starting testing python function ..."

python main.py --test_model \
--model_name=Ours \
--dataset_path=/ibex/projects/c2006/KITTI/tracking/training/ \
--search_space=Kalman \
--reference_BB=previous_result \
--number_candidate=$SLURM_ARRAY_TASK_ID

echo "...testing function Done"



