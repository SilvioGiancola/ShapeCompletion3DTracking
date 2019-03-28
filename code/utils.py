import torch
import os
import copy

import numpy as np
from external.python_plyfile.plyfile import PlyElement, PlyData
from pyquaternion import Quaternion

from data_classes import PointCloud
from metrics import estimateOverlap


def distanceBB_Gaussian(box1, box2, sigma=1):
    off1 = np.array([
        box1.center[0], box1.center[2],
        Quaternion(matrix=box1.rotation_matrix).degrees
    ])
    off2 = np.array([
        box2.center[0], box2.center[2],
        Quaternion(matrix=box2.rotation_matrix).degrees
    ])
    dist = np.linalg.norm(off1 - off2)
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return score


# IoU or Gaussian score map
def getScoreGaussian(offset, sigma=1):
    coeffs = [1, 1, 1 / 5]
    dist = np.linalg.norm(np.multiply(offset, coeffs))
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return torch.tensor([score])


def getScoreIoU(a, b):
    score = estimateOverlap(a, b)
    return torch.tensor([score])


def getScoreHingeIoU(a, b):
    score = estimateOverlap(a, b)
    if score < 0.5:
        score = 0.0
    return torch.tensor([score])


def getOffsetBB(box, offset):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    # REMOVE TRANSfORM
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
    new_box.translate(np.array([offset[0], offset[1], 0]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def voxelize(PC, dim_size=[48, 108, 48]):
    # PC = normalizePC(PC)
    if np.isscalar(dim_size):
        dim_size = [dim_size] * 3
    dim_size = np.atleast_2d(dim_size).T
    PC = (PC + 0.5) * dim_size
    # truncate to integers
    xyz = PC.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dim_size), 0)
    xyz = xyz[:, valid_ix]
    out = np.zeros(dim_size.flatten(), dtype=np.float32)
    out[tuple(xyz)] = 1
    # print(out)
    return out


def regularizePC2(model, PC):
    return regularizePC(PC=PC, model=model)


def regularizePC(PC, model):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != model.input_size:
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=model.input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape(1, 3, model.input_size)

    else:
        PC = np.zeros((1, 3, model.input_size))

    return torch.from_numpy(PC).float()


def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):

    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))
    # print(points)

    for PC, box in zip(PCs, boxes):
        cropped_PC = cropAndCenterPC(
            PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

    PC = PointCloud(points)

    return PC


def cropPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC


def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC


def distanceBB(box1, box2):

    eucl = np.linalg.norm(box1.center - box2.center)
    angl = Quaternion.distance(
        Quaternion(matrix=box1.rotation_matrix),
        Quaternion(matrix=box2.rotation_matrix))
    return eucl + angl


def generate_boxes(box, search_space=[[0, 0, 0]]):
    # Geenrate more candidate boxes based on prior and search space
    # Input : Prior position, search space and seaarch size
    # Output : List of boxes

    candidate_boxes = [getOffsetBB(box, offset) for offset in search_space]
    return candidate_boxes


def getDataframeGT(anno):
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": anno["alpha"],
        "bbox_left": anno["bbox_left"],
        "bbox_top": anno["bbox_top"],
        "bbox_right": anno["bbox_right"],
        "bbox_bottom": anno["bbox_bottom"],
        "height": anno["height"],
        "width": anno["width"],
        "length": anno["length"],
        "x": anno["x"],
        "y": anno["y"],
        "z": anno["z"],
        "rotation_y": anno["rotation_y"]
    }
    return df


def getDataframe(anno, box, score):
    myquat = (box.orientation * Quaternion(axis=[1, 0, 0], radians=-np.pi / 2))
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": 0.0,
        "bbox_left": 0.0,
        "bbox_top": 0.0,
        "bbox_right": 0.0,
        "bbox_bottom": 0.0,
        "height": box.wlh[2],
        "width": box.wlh[0],
        "length": box.wlh[1],
        "x": box.center[0],
        "y": box.center[1] + box.wlh[2] / 2,
        "z": box.center[2],
        "rotation_y":
        np.sign(myquat.axis[1]) * myquat.radians,  # this_anno["rotation_y"], #
        "score": score
    }
    return df


def saveTrackingResults(df_3D, dataset_loader, export=None, epoch=-1):

    for i_scene, scene in enumerate(df_3D.scene.unique()):
        new_df_3D = df_3D[df_3D["scene"] == scene]
        new_df_3D = new_df_3D.drop(["scene"], axis=1)
        new_df_3D = new_df_3D.sort_values(by=['frame', 'track_id'])

        os.makedirs(os.path.join("results", export, "data"), exist_ok=True)
        if epoch == -1:
            path = os.path.join("results", export, "data", f"{scene}.txt")
        else:
            path = os.path.join("results", export, "data",
                                f"{scene}_epoch{epoch}.txt")

        new_df_3D.to_csv(
            path, sep=" ", header=False, index=False, float_format='%.6f')


def write_ply(PC, filename):
    PC = PC.T
    PC = [tuple(element) for element in PC]
    el = PlyElement.describe(
        np.array(PC, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
    PlyData([el]).write(filename)


def write_ply_tensor(tensor, filename):
    tensor = tensor.t().cpu().numpy()
    tensor = [tuple(element) for element in tensor]
    el = PlyElement.describe(
        np.array(tensor, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
        'vertex')
    PlyData([el]).write(filename)
