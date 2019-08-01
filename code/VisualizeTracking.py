from pyquaternion import Quaternion
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import torch
import imageio

from NNmodel import Model
from Dataset import SiameseTest
from utils import generate_boxes, getModel
from utils import cropAndCenterPC, regularizePC
from data_classes import Box, PointCloud

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D


def main(args):

    # Plotting parameters
    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (15, 5),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'figure.max_open_warning': 1000
    }

    pylab.rcParams.update(params)

    # Create Model
    if "Random" in args.model_name:
        AE_chkpt_file = ""
        chkpt_file = None
    if "PreTrained" in args.model_name:
        AE_chkpt_file = os.path.join("..", "models", "Pretrain_ShapeNet",
                                     "model.pth.tar")
        chkpt_file = None
    if "Ours" in args.model_name:
        AE_chkpt_file = os.path.join("..", "models", "Pretrain_ShapeNet",
                                     "model.pth.tar")
        chkpt_file = os.path.join("..", "models", "Ours", "model.pth.tar")

    model = Model(
        128, chkpt_file=chkpt_file, AE_chkpt_file=AE_chkpt_file).cuda()
    model.eval()

    # Create Dataset
    dataset = SiameseTest(
        model=model, path=args.path_KITTI, split="Test", scale_BB=1.25)

    # Create search space
    x_space = np.linspace(-4, 4, 9)  # min, max, number of sample
    y_space = np.linspace(-4, 4, 9)  # min, max, number of sample
    a_space = np.linspace(-10, 10, 3)  # min, max, number of sample
    X, Y, A = np.meshgrid(x_space, y_space, a_space)  # create mesh grid
    dataset.search_grid = np.array([X.flatten(), Y.flatten(), A.flatten()]).T
    dataset.num_candidates_perframe = len(dataset.search_grid)

    # Get List of PCs and BBs from dataset
    PCs, BBs, list_of_anno = dataset[args.track]

    # Loop over Point Cloud and Bounding Boxes
    for i in tqdm(range(1, len(PCs))):

        this_PC = PCs[i]
        this_BB = BBs[i]
        ref_BB = BBs[i]

        # create candidate PCs and BBs
        search_space = dataset.search_grid
        candidate_BBs = generate_boxes(ref_BB, search_space=search_space)
        candidate_PCs = [
            cropAndCenterPC(
                this_PC,
                box,
                offset=dataset.offset_BB,
                scale=dataset.scale_BB,
                normalize="PreTrained" in args.model_name)
            for box in candidate_BBs
        ]
        candidate_PCs_reg = [regularizePC(PC, model) for PC in candidate_PCs]
        candidate_PCs_torch = torch.cat(candidate_PCs_reg, dim=0).cuda()

        # create model PC
        model_PC = getModel(
            PCs[:i],
            BBs[:i],
            offset=dataset.offset_BB,
            scale=dataset.scale_BB,
            normalize="PreTrained" in args.model_name)

        repeat_shape = np.ones(len(candidate_PCs_torch.shape), dtype=np.int32)
        repeat_shape[0] = len(candidate_PCs)
        model_PC_encoded = regularizePC(model_PC, model).repeat(
            tuple(repeat_shape)).cuda()

        # infer model
        output, model_PC_decoded = model(candidate_PCs_torch, model_PC_encoded)
        model_PC_decoded = PointCloud(
            model_PC_decoded.detach().cpu().numpy()[0])
        if "PreTrained" in args.model_name:
            normalizer = [ref_BB.wlh[1], ref_BB.wlh[0], ref_BB.wlh[2]]
            model_PC_decoded.points = model_PC_decoded.points * np.atleast_2d(
                normalizer).T

        # store scores for all candidates
        scores = output.detach().cpu().numpy()
        idx = np.argmax(scores)  # select index of higest score
        box = candidate_BBs[idx]  # select box with highest score

        # keep highest score for angle space
        scores0 = scores[1::3]  # angle = 0
        scoresp10 = scores[2::3]  # angle = 10
        scoresm10 = scores[0::3]  # angle = -10
        scores = np.max(np.stack([scores0, scoresp10, scoresm10]), axis=0)

        X, Y = np.meshgrid(x_space, y_space)
        Z = scores.reshape(len(x_space), len(y_space))

        view_PC = cropAndCenterPC(this_PC, this_BB, offset=5)
        view_BB = Box([0, 0, 0], this_BB.wlh, Quaternion())

        # Crop point clouds around GT
        x_filt_max = view_PC.points[0, :] < 5
        x_filt_min = view_PC.points[0, :] > -5
        y_filt_max = view_PC.points[1, :] < 5
        y_filt_min = view_PC.points[1, :] > -5
        z_filt_max = view_PC.points[2, :] < 2
        z_filt_min = view_PC.points[2, :] > -1
        close = np.logical_and(x_filt_min, x_filt_max)
        close = np.logical_and(close, y_filt_min)
        close = np.logical_and(close, y_filt_max)
        close = np.logical_and(close, z_filt_min)
        close = np.logical_and(close, z_filt_max)
        view_PC = PointCloud(view_PC.points[:, close])

        # Create figure for TRACKING
        fig = plt.figure(figsize=(15, 10), facecolor="white")
        # Create axis in 3D
        ax = fig.gca(projection='3d')

        # Scatter plot the cropped point cloud
        ratio = 1
        ax.scatter(
            view_PC.points[0, ::ratio],
            view_PC.points[1, ::ratio],
            view_PC.points[2, ::ratio] / 2 - 1,
            s=3,
            c=view_PC.points[2, ::ratio])

        # point order to draw a full Box
        order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

        # Plot best results
        ax.plot(
            view_BB.corners()[0, order],
            view_BB.corners()[1, order],
            view_BB.corners()[2, order] / 2 - 1,
            color="red",
            alpha=0.5,
            linewidth=5,
            linestyle=":")

        # center the box to plot the GT
        box.translate(-this_BB.center)
        box.rotate(this_BB.orientation.inverse)

        # Plot GT box
        ax.plot(
            box.corners()[0, order],
            box.corners()[1, order],
            box.corners()[2, order] / 2 - 1,
            color="blue",
            alpha=0.5,
            linewidth=2,
            linestyle=":")

        # point order to draw the visible part of the box
        # order = [3, 0, 1, 2, 3, 7, 4, 0, 4, 5, 1]
        order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]

        # Plot best results
        ax.plot(
            view_BB.corners()[0, order],
            view_BB.corners()[1, order],
            view_BB.corners()[2, order] / 2 - 1,
            color="red",
            linewidth=5)

        # Plot GT box
        ax.plot(
            box.corners()[0, order],
            box.corners()[1, order],
            box.corners()[2, order] / 2 - 1,
            color="blue",
            linewidth=2)

        # Plot results for all cadidate as a surface
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.coolwarm,
            rstride=1,
            cstride=1,
            linewidth=0.3,
            edgecolors=[0, 0, 0, 0.8],
            antialiased=True,
            vmin=0,
            vmax=1)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.2))
        ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.2))
        ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.2))

        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        ax.set_xticks(x_space[::2])
        ax.set_yticks(y_space[::2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.view_init(20, -140)

        plt.tight_layout()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-1.5, 1)

        # Save figure as Tracking results
        os.makedirs(
            os.path.join(args.path_results, f"{args.track:04.0f}", "Tracking"),
            exist_ok=True)
        plt.savefig(
            os.path.join(args.path_results, f"{args.track:04.0f}",
                         "Tracking", f"{i}_{args.model_name}.png"),
            format='png',
            dpi=100)



        # Create figure for RECONSTRUCTION
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        # Create axis in 3D
        ax = fig.gca(projection='3d')

        # select 2K point to visualize
        sample = np.random.randint(
            low=0,
            high=model_PC_decoded.points.shape[1],
            size=2048,
            dtype=np.int64)
        # Scatter plot the point cloud
        ax.scatter(
            model_PC_decoded.points[0, sample],
            model_PC_decoded.points[1, sample],
            model_PC_decoded.points[2, sample],
            s=3,
            c=model_PC_decoded.points[2, sample])

        # Plot the car BB
        order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
        ax.plot(
            view_BB.corners()[0, order],
            view_BB.corners()[1, order],
            view_BB.corners()[2, order],
            color="red",
            alpha=0.5,
            linewidth=3,
            linestyle=":")
        order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]
        ax.plot(
            view_BB.corners()[0, order],
            view_BB.corners()[1, order],
            view_BB.corners()[2, order],
            color="red",
            linewidth=3)

        # setup axis
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels([], fontsize=10)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([], fontsize=10)
        ax.set_zticks([-1, 0, 1])
        ax.set_zticklabels([], fontsize=10)
        ax.view_init(20, -140)
        plt.tight_layout()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1.5, 1.5)

        # Save figure as Decoded Model results
        os.makedirs(
            os.path.join(args.path_results, f"{args.track:04.0f}",
                         "Reconstruction"),
            exist_ok=True)
        plt.savefig(
            os.path.join(args.path_results, f"{args.track:04.0f}",
                         "Reconstruction", f"{i}_{args.model_name}.png"),
            format='png',
            dpi=100)



        # Create figure for MODEL PC
        fig = plt.figure(figsize=(9, 6), facecolor="white")
        ax = fig.gca(projection='3d')


        if "PreTrained" in args.model_name:
            model_PC = getModel(
                PCs[:i],
                BBs[:i],
                offset=dataset.offset_BB,
                scale=dataset.scale_BB)

        # sample 2K Points
        sample = np.random.randint(
            low=0, high=model_PC.points.shape[1], size=2048, dtype=np.int64)
        # Scatter plot the Point cloud
        ax.scatter(
            model_PC.points[0, sample],
            model_PC.points[1, sample],
            model_PC.points[2, sample],
            s=3,
            c=model_PC.points[2, sample])

        # Plot the Bounding Box
        order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
        ax.plot(
            view_BB.corners()[0, order],
            view_BB.corners()[1, order],
            view_BB.corners()[2, order],
            color="red",
            alpha=0.5,
            linewidth=3,
            linestyle=":")
        order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]
        ax.plot(
            view_BB.corners()[0, order],
            view_BB.corners()[1, order],
            view_BB.corners()[2, order],
            color="red",
            linewidth=3)

        # setup axis
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels([], fontsize=10)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([], fontsize=10)
        ax.set_zticks([-1, 0, 1])
        ax.set_zticklabels([], fontsize=10)
        ax.view_init(20, -140)
        plt.tight_layout()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1.5, 1.5)

        # Save figure as Model results
        os.makedirs(
            os.path.join(args.path_results, f"{args.track:04.0f}", "Model"),
            exist_ok=True)
        plt.savefig(
            os.path.join(args.path_results, f"{args.track:04.0f}", "Model",
                         f"{i}_{args.model_name}.png"),
            format='png',
            dpi=100)



    # create GIF Tracking
    images = []
    for i in tqdm(range(1, len(PCs))):
        image_path = os.path.join(args.path_results, f'{args.track:04.0f}',
                                  "Tracking", f'{i}_{args.model_name}.png')
        images.append(imageio.imread(image_path))

    image_path = os.path.join(args.path_results,
                     f'{args.track:04.0f}_{args.model_name}_Tracking.gif')
    imageio.mimsave(image_path, images)

    # create GIF GT Car
    images = []
    for i in tqdm(range(1, len(PCs))):
        image_path = os.path.join(args.path_results, f'{args.track:04.0f}',
                                  "Model", f'{i}_{args.model_name}.png')
        images.append(imageio.imread(image_path))

    image_path = os.path.join(
        args.path_results, f'{args.track:04.0f}_{args.model_name}_Model.gif')
    imageio.mimsave(image_path, images)

    # create GIF Reconstructed Car
    images = []
    for i in tqdm(range(1, len(PCs))):
        image_path = os.path.join(args.path_results, f'{args.track:04.0f}',
                                  "Reconstruction",
                                  f'{i}_{args.model_name}.png')
        images.append(imageio.imread(image_path))

    image_path = os.path.join(
        args.path_results,
        f'{args.track:04.0f}_{args.model_name}_Reconstruction.gif')
    imageio.mimsave(image_path, images)



if __name__ == '__main__':

    parser = ArgumentParser(
        description='Visualize Tracking Results',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--GPU',
        required=False,
        type=int,
        default=-1,
        help='ID of the GPU to use')
    parser.add_argument(
        '--model_name',
        required=False,
        type=str,
        default="Ours",
        help='model to infer (Random/PreTrained/Ours)')
    parser.add_argument(
        '--track',
        required=False,
        type=int,
        default=29,
        help='track to infer (supp. mat. are 29/45/91)')
    parser.add_argument(
        '--path_results',
        required=False,
        type=str,
        default=os.path.join("..", "results"),
        help='path to save the results')
    parser.add_argument(
        '--path_KITTI',
        required=False,
        type=str,
        default="KITTI/tracking/training",
        help='path for the KITTI dataset')

    args = parser.parse_args()

    # Select GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # Enforce reproducibility in torch
    torch.manual_seed(1155)

    # start main
    main(args)
