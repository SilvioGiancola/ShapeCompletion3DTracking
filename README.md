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


## Download KITTI Tracking dataset

Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

You will need to download the data for
[velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), 
[calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and
[label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).

Place the 3 folders in the same parent folder as following:
```
[Parent Folder]
--> [calib]
    --> {0000-0020}.txt
--> [label_02]
    --> {0000-0020}.txt
--> [velodyne]
    --> [0000-0020] folders with velodynes .bin files
```



## Create Environment

```
conda create -y -n ShapeCompletion3DTracking python tqdm numpy pandas shapely matplotlib pomegranate
source activate ShapeCompletion3DTracking
conda install -y pytorch=0.4.1 cuda90 -c pytorch
pip install pyquaternion 
```


## Train a model

`python main.py --train_model --model_name=<Name of your model> --dataset_path=<Path to KITTI Tracking folder>` 


```
OPT: 
    --model_name=<Name of your model>
    --dataset_path=<Path to KITTI Tracking>
    --GPU=1: enforce the use of GPU 1 
    --tiny: use a tiny set of KITTI Tracking
```

## Test a model

`python main.py --test_model --model_name=<Name of your model> --dataset_path=<Path to KITTI Tracking folder>` 


```
OPT: 
    --model_name=<Name of your model>
    --dataset_path=<Path to KITTI Tracking>
    --GPU=1: enforce the use of GPU 1 
    --tiny: use a tiny set of KITTI Tracking
```