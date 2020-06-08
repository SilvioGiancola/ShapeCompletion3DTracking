import warnings
import numpy as np
import os
import os.path as osp
import re
import torch
import torch.utils.data
import copy
import pickle

from external.python_plyfile.plyfile import PlyElement, PlyData

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}


def snc_category_to_synth_id():
    d = snc_synth_id_to_category
    inv_map = {v: k for k, v in d.items()}
    return inv_map


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

def save_text_arr(file_name, arr):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    with open(file_name, 'w') as f:
        for element in arr:
            f.write(str(element)+"\n")


def read_text_arr(file_name):
    '''Restore data previously saved with pickle_data().
    '''
    with open(file_name, 'r') as f:
        arr = [int(element[:-1]) for element in f]
    return arr


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']])
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def pc_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme: 
    i.e. /syn_id/model_name.ply
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id


def load_all_point_clouds_under_folder(top_dir, file_ending='.ply', verbose=False):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    return file_names

def voxelize(pcloud, dim_size):
    pc_max = np.atleast_2d(pcloud.max(1)).T
    pc_min = np.atleast_2d(pcloud.min(1)).T
    pcloud = pcloud - pc_min
    pcloud = np.true_divide(pcloud,pc_max - pc_min)
    if np.isscalar(dim_size):
        dim_size = [dim_size]*3
    dim_size = np.atleast_2d(dim_size).T
    pcloud = pcloud*dim_size
    # truncate to integers
    xyz = pcloud.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dim_size), 0)
    xyz = xyz[:,valid_ix]
    out = np.zeros(dim_size.flatten(), dtype=np.float32)
    out[tuple(xyz)] = 1
    return out


class AEDataSet(torch.utils.data.Dataset):

    def __init__(self, root, class_name, missing_data=False, dim_size=[48,108,48], data_representation = "pointcloud"):
        syn_id = snc_category_to_synth_id()[class_name]
        class_dir = osp.join(root , syn_id)
        self.pclouds = load_all_point_clouds_under_folder(class_dir, file_ending='.ply', verbose=True)
        self.dim_size=dim_size
        self.missing_data=missing_data
        self.data_representation=data_representation

    def __getitem__(self, index):
        pcloud, model_id, synet_id = pc_loader(self.pclouds[index])
        target = copy.deepcopy(pcloud)
        if self.missing_data:
            num_pts = len(pcloud[0])
            indices = range(num_pts)
            missing_points = np.int((1 - np.random.beta(a=2, b=5))*num_pts)
            missing_indices = np.random.choice(indices, size=missing_points, replace=False)
            pcloud[:,missing_indices] = 0
        if(self.data_representation=="pointcloud"):
            voxels = np.array([voxelize(pcloud,self.dim_size)])
            target = np.array([voxelize(target,self.dim_size)])
            return voxels, target
        elif(self.data_representation=="voxel"):
            return pcloud, target
        else:
            print("Invalida data representation")
            return pcloud, target

    def __len__(self):
        return len(self.pclouds)

