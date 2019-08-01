import os
import time
import logging

from datetime import datetime

import torch
import torch.cuda
import torch.optim
import torch.utils.data

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from NNmodel import Model
from train import trainer, test
from Dataset import SiameseTrain, SiameseTest
from PCLosses import ChamferLoss
from metrics import AverageMeter
import numpy as np

# for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    if args.test_model and not args.train_model:
        chkpt_file = os.path.join("..", "models", args.model_name,
                                  "model.pth.tar")
    else:
        chkpt_file = None

    model = Model(args.bneck_size, AE_chkpt_file=args.finetune, chkpt_file=chkpt_file).cuda()
    logging.info(model)

    split_train = "Train"
    split_valid = "Valid"
    split_test = "Test"
    if args.tiny:
        split_train = "Tiny_" + split_train
        split_valid = "Tiny_" + split_valid
        split_test = "Tiny_" + split_test

    # define DATASETS
    if args.train_model:
        dataset_Training = SiameseTrain(
            model=model,
            path=args.dataset_path,
            split=split_train,
            category_name=args.category_name,
            regress=args.regress,
            sigma_Gaussian=args.sigma_Gaussian,
            offset_BB=args.offset_BB,
            scale_BB=args.scale_BB)

        dataloader_Training = torch.utils.data.DataLoader(
            dataset_Training,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.max_num_worker,
            pin_memory=True)

        dataset_Validation = SiameseTrain(
            model=model,
            path=args.dataset_path,
            split=split_valid,
            category_name=args.category_name,
            regress=args.regress,
            sigma_Gaussian=args.sigma_Gaussian,
            offset_BB=args.offset_BB,
            scale_BB=args.scale_BB)

        dataloader_Validation = torch.utils.data.DataLoader(
            dataset_Validation,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_num_worker,
            pin_memory=True)

    if args.test_model:
        dataset_Test = SiameseTest(
            model=model,
            path=args.dataset_path,
            split=split_test,
            category_name=args.category_name,
            offset_BB=args.offset_BB,
            scale_BB=args.scale_BB)

        test_loader = torch.utils.data.DataLoader(
            dataset_Test,
            collate_fn=lambda x: x,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)

    # define criterions
    criterion_tracking = torch.nn.MSELoss()
    criterion_completion = ChamferLoss()

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-04,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False)

    # define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=3)

    if args.train_model:
        trainer(
            dataloader_Training,
            dataloader_Validation,
            model,
            optimizer,
            scheduler,
            criterion_tracking,
            criterion_completion,
            model_name=args.model_name,
            lambda_completion=args.lambda_completion)

    if args.test_model:

        Success_run = AverageMeter()
        Precision_run = AverageMeter()

        if dataset_Test.isTiny():
            max_epoch = 25
        else:
            max_epoch = 5

        for epoch in range(max_epoch):
            Succ, Prec = test(
                test_loader,
                model,
                epoch=epoch + 1,
                model_name=args.model_name,
                shape_aggregation=args.shape_aggregation,
                search_space=args.search_space,
                number_candidate=args.number_candidate,
                reference_BB=args.reference_BB,
                model_fusion=args.model_fusion,
                IoU_Space=args.IoU_Space,
                DetailedMetrics=args.detailed_metrics)
            Success_run.update(Succ)
            Precision_run.update(Prec)
            logging.info(
                f"mean Succ/Prec {Success_run.avg}/{Precision_run.avg}")


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Train or test Shape Completion for 3D Tracking',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test_model', required=False, action='store_true')
    parser.add_argument('--train_model', required=False, action='store_true')
    parser.add_argument(
        '--detailed_metrics', required=False, action='store_true')

    parser.add_argument(
        '--model_name',
        required=False,
        type=str,
        default="myModel",
        help='Where to export the model')

    parser.add_argument(
        '--dataset_path',
        required=False,
        type=str,
        default='/home/zarzarj/CVPR2019/Kitti/2DTracking/training',
        help='dataset Path')

    parser.add_argument('--tiny', required=False, action='store_true')



    parser.add_argument(
        '--bneck_size',
        required=False,
        type=int,
        default=128,
        help='Size of the bottleneck')

    parser.add_argument(
        '--batch_size', required=False, type=int, default=32, help='Batch size')

    parser.add_argument(
        '--GPU',
        required=False,
        type=int,
        default=-1,
        help='ID of the GPU to use')

    parser.add_argument(
        '--model_fusion',
        required=False,
        type=str,
        default="pointcloud",
        help='early or late fusion (pointcloud/latent/space)')

    parser.add_argument(
        '--shape_aggregation',
        required=False,
        type=str,
        default="",
        help='Aggregation of shapes (first/previous/firstandprevious/all/AVG/MEDIAN/MAX)')



    parser.add_argument(
        '--search_space',
        required=False,
        type=str,
        default='Exhaustive',
        help='Search space (Exhaustive/Kalman/Particle/GMM<N>)')

    parser.add_argument(
        '--number_candidate',
        required=False,
        type=int,
        default=125,
        help='Number of candidate for Kalman, Particle or GMM search space')

    parser.add_argument(
        '--reference_BB',
        required=False,
        type=str,
        default="current_gt",
        help='previous_result/previous_gt/current_gt')



    parser.add_argument(
        '--regress',
        required=False,
        type=str,
        default='gaussian',
        help='how to regress (IoU/Gaussian)')

    parser.add_argument(
        '--category_name',
        required=False,
        type=str,
        default='Car',
        help='Object to Track (Car/Pedetrian/Van/Cyclist)')



    parser.add_argument(
        '--lambda_completion',
        required=False,
        type=float,
        default=1e-6,
        help='lambda ratio for completion loss')

    parser.add_argument(
        '--sigma_Gaussian',
        required=False,
        type=float,
        default=1,
        help='Gaussian distance variation sigma for regression in training')

    parser.add_argument(
        '--offset_BB',
        required=False,
        type=float,
        default=0,
        help='offset around the BB in meters')
    parser.add_argument(
        '--scale_BB',
        required=False,
        type=float,
        default=1.25,
        help='scale of the BB before cropping')

    parser.add_argument(
        '--IoU_Space',
        required=False,
        type=int,
        default=3,
        help='IoUBox vs IoUBEV (2 vs 3)')



    parser.add_argument(
        '--max_num_worker',
        required=False,
        type=int,
        default=6,
        help='number of worker')

    parser.add_argument(
        '--loglevel',
        required=False,
        type=str,
        default='INFO',
        help='logging level')

    parser.add_argument(
        '--models_path',
        required=False,
        type=str,
        default=os.path.join("..", "models"),
        help='Folder for models to be stored or read from.')
    parser.add_argument(
        '--model_path',
        required=False,
        type=str,
        default=os.path.join("..", "models", "Pretrain_ShapeNet"),
        help='pre-trained model')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join(args.models_path, args.model_name), exist_ok=True)
    loggerpath = os.path.join(args.models_path, args.model_name, "logging.log")

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.models_path, args.model_name,
                             datetime.now().strftime('%Y-%m-%d %H-%M-%S.log'))),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    if(args.model_path is not None and args.model_path is not ""):
        args.finetune = os.path.join(args.model_path, "model.pth.tar")
    else:
        args.finetune = ""

    start = time.time()
    main(args)
    logging.info(
        'Total Execution Time is {0} seconds'.format(time.time() - start))
