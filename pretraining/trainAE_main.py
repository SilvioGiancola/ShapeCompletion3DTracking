import os
import os.path as osp

import importlib

import numpy as np

import torch
import torch.cuda
import torch.optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler

from train import trainer

from PCLosses import ChamferLoss
from AEModel import PCAutoEncoder, VoxelAutoEncoder

from AEDataSet import AEDataSet, save_text_arr, read_text_arr, create_dir

# for reproducibility
torch.manual_seed(0)

# a utility function to print the progress of a for-loop 
# it is recomended that you install `tqdm` package
def _verbosify(iterable):
    # shows only the iteration number and how many iterations are left
    try:
        len_iterable = len(iterable)
    except Exception:
        len_iterable = None
    for i, element in enumerate(iterable, 1):
        print('\rIteration # ')
        print(i)
        yield element

def verbosify(iterable, **kwargs):
    # try to use tqdm (shows the speed and the remaining time left)
    if importlib.util.find_spec('tqdm') is not None:
        tqdm = importlib.import_module('tqdm').tqdm
        if 'file' not in kwargs:
            kwargs['file'] = importlib.import_module('sys').stdout
        if 'leave' not in kwargs:
            kwargs['leave'] = False
        return tqdm(iterable, **kwargs)
    else:
        return iter(_verbosify(iterable))


def main(args):



    top_out_dir = 'data/'          # Use to save Neural-Net check-points etc.
    top_in_dir = args.top_in_dir  # Top-dir of where point-clouds are stored. (ShapeNet)

    experiment_name = 'single_class_ae'
    n_pc_points = 2048                # Number of points per model.
    bneck_size = args.bneck_size                  # Bottleneck-AE size
    ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
    class_name = 'car'

    data_representation = args.data_representation

    dataset = AEDataSet(top_in_dir, class_name, missing_data=args.missing_data)
    
    if(data_representation=="pointcloud"):
        model = PCAutoEncoder(bneck_size=bneck_size).cuda()
        criterion = ChamferLoss()
        export = "model/PCAE/"
    elif data_representation =="voxel":
        model = VoxelAutoEncoder(bneck_size=bneck_size).cuda()
        criterion = torch.nn.BCELoss()
        export="model/Voxel/"
    else:
        print("Invalid data representation")
    os.makedirs(export, exist_ok=True)

    dataset_length = len(dataset)

    print("BottleNeck size set to: " + str(bneck_size))


    try:
        train_set = read_text_arr("train_set.txt")
        valid_set = read_text_arr("valid_set.txt")
        test_set = read_text_arr("test_set.txt")
    except:
        train_frac = 0.8
        valid_frac = 0.1
        test_frac = 1 - train_frac - valid_frac

        train_length = int(dataset_length * train_frac)
        valid_length = int(dataset_length * valid_frac)
        test_length = dataset_length - train_length - valid_length

        ## define our indices -- our dataset has 9 elements and we want a 8:4 split
        num_train = len(dataset)
        indices = range(num_train)

        # Random, non-contiguous split
        train_set = np.random.choice(indices, size=train_length, replace=False)
        not_train_set = list(set(indices) - set(train_set))
        valid_set = np.random.choice(not_train_set, size=valid_length, replace=False)
        test_set = list(set(indices) - set(train_set) - set(valid_set))

        save_text_arr("train_set.txt", train_set)
        save_text_arr("valid_set.txt", valid_set)
        save_text_arr("test_set.txt", test_set)

    train_sampler = SubsetRandomSampler(train_set)
    valid_sampler = SubsetRandomSampler(valid_set)
    test_sampler = SubsetRandomSampler(test_set)
        

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=50, shuffle=False,
        num_workers=4, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=50, shuffle=False,
        num_workers=4, pin_memory=True, sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=50, shuffle=False,
        num_workers=4, pin_memory=True, sampler=test_sampler)

    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, 
                                 betas=(0.9, 0.999), eps=1e-08, 
                                 weight_decay=0, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=3)



    best_chk = "model/PCAE_bneck" + str(bneck_size) +"_best.pth.tar"
    trainer(train_loader, val_loader, test_loader, 
            model, optimizer, scheduler, criterion, 
            epochs=100, export=export, data_representation=data_representation, infer="model/Output_missing_data/", )



from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd

if __name__ == '__main__':


    parser = ArgumentParser(description='Train a PCAE', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bneck_size', required=False, type=int,   default=128,    help='Size of the bottleneck' )
    parser.add_argument('--data_representation', required=False, type=str,   default="voxel",    help='Data representation used for AE, "voxel" or "pointcloud"' )
    parser.add_argument('--missing_data', required=False, type=bool,   default=False,    help='Whether to set a random number of points to 0 when loading point clouds' )
    parser.add_argument('--jobid',      required=False, type=int,   default=-1,     help='Job ID for batch in Ibex' )
    parser.add_argument('--csv_file',   required=False, type=str,   default="",     help='Csv file to load arguments from' )
    parser.add_argument('--top_in_dir', required=True, type=str, help='Shapenet top dir' )

    args = parser.parse_args()

    if(".csv" in args.csv_file and args.jobid >= 0):        
        Params = pd.read_csv(args.csv_file).iloc[args.jobid]
        args.bneck_size = Params.bneck_size

    main(args)
