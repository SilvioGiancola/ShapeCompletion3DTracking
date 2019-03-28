# Leveraging Shape Completion for 3D Siamese Tracking
Supplementary Code for the CVPR'19 paper entitled [Leveraging Shape Completion for 3D Siamese Tracking](https://arxiv.org/pdf/1903.01784.pdf)

[![Supplementary Video](https://img.youtube.com/vi/2-NAaWSSrGA/0.jpg)](https://www.youtube.com/watch?v=2-NAaWSSrGA "Supplementary Video")

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

## Download KITTI Tracking dataset

Download the datset from [here](http://www.cvlibs.net/datasets/kitti/).



## Create Environment

```
conda create -y -n ShapeCompletion3DTracking python tqdm numpy pandas shapely matplotlib pomegranate
source activate ShapeCompletion3DTracking
conda install -y pytorch=0.4.1 cuda90 -c pytorch
pip install pyquaternion 
```


## Train a model

`python main.py --train_model --model_name=<Name of your model> --dataset_path=<Path to KITTI Tracking>` 


```
OPT: 
    --model_name=<Name of your model>
    --dataset_path=<Path to KITTI Tracking>
    --GPU=1: enforce the use of GPU 1 
    --tiny: use a tiny set of KITTI Tracking
```

## Test a model

`python main.py --test_model --model_name=<Name of your model> --dataset_path=<Path to KITTI Tracking>` 


```
OPT: 
    --model_name=<Name of your model>
    --dataset_path=<Path to KITTI Tracking>
    --GPU=1: enforce the use of GPU 1 
    --tiny: use a tiny set of KITTI Tracking
```