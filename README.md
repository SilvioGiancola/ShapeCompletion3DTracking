# Leveraging Shape Completion for 3D Siamese Tracking
Supplementary Code for the CVPR'19 paper entitled: "Leveraging Shape Completion for 3D Siamese Tracking"

## Citation

```
@InProceedings{Giancola_2018_CVPR,
author = {Giancola, Silvio and Zarzar, Jesus and Ghanem, Bernard},
title = {Leverage Shape Completion for 3D Siamese Tracking},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```


# Usage

`Still under construction`

## Create Environment

```
conda create -y -n ShapeCompletion3DTracking python tqdm numpy pandas shapely matplotlib pomegranate
source activate ShapeCompletion3DTracking
conda install -y pytorch=0.4.1 cuda90 -c pytorch
pip install pyquaternion 
```

## Download KITTI Tracking dataset


## Train a model

`python main.py --model_name=myModel --dataset_path=<Path to KITTI Tracking> --train_model` 


```
OPT: 
    --model_name=<Name of your model>
    --dataset_path=<Path to KITTI Tracking>
    --GPU=1: enforce the use of GPU 1 
    --tiny: use a tiny set of KITTI
```

## Test a model

`python main.py --model_name=myModel --dataset_path=<Path to KITTI Tracking> --test_model` 


```
OPT: 
    --model_name=<Name of your model>
    --dataset_path=<Path to KITTI Tracking>
    --GPU=1: enforce the use of GPU 1 
    --tiny: use a tiny set of KITTI
```