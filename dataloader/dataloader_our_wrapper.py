import glob
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, motions, sparses, all_info, input_motion_length=196,
                 train_dataset_repeat_times=1, normalization=True, ):
        self.motions = motions
        self.sparses = sparses
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.input_motion_length = input_motion_length

        self.motion_54 = []
        self.up_idx = [15, 20, 21]
        self.n_joint = 22
        for idx in range(len(self.sparses)):
            sparse = self.sparses[idx]
            seq = sparse.shape[0]
            # # rot_absolute = self.motions[idx].reshape(-1, self.n_joint, 6)
            # rot_abs = sparse[:, :self.n_joint * 6].reshape(-1, self.n_joint, 6)
            # rot_vel = sparse[:, self.n_joint * 6: self.n_joint * 12].reshape(-1, self.n_joint, 6)
            # pos = sparse[:, self.n_joint * 12: self.n_joint * 15].reshape(-1, self.n_joint, 3)
            # pos_vel = sparse[:, self.n_joint * 15: self.n_joint * 18].reshape(-1, self.n_joint, 3)
            # # sparse_396 = torch.cat((rot_absolute, rot_vel, pos, pos_vel), dim=-1).reshape(seq, -1)  # (seq, 22, 18)
            # sparse_396_abs = torch.cat((rot_abs, rot_vel, pos, pos_vel), dim=-1)
            # self.motion_54.append(sparse_396_abs[:, self.up_idx])
            self.motion_54.append(sparse.reshape(-1, 3, 18))

    def __len__(self):
        return len(self.motions) * self.train_dataset_repeat_times

    def __getitem__(self, idx):
        sparse = self.motion_54[idx % len(self.motions)].float()  # (bs, seq, 3, 18)
        motion = self.motions[idx % len(self.motions)].float()
        seqlen = motion.shape[0]

        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]
        motion = motion[idx: idx + self.input_motion_length]  # 随机切出一个长度为196帧的序列
        sparse = sparse[idx: idx + self.input_motion_length]

        return motion, sparse


class TestDataset(Dataset):
    def __init__(self, all_info, filename_list):
        self.filename_list = filename_list
        self.motions = []
        self.sparses = []
        self.body_params = []
        self.head_motion = []
        for i in all_info:
            self.motions.append(i["rotation_local_full_gt_list"])
            self.sparses.append(i["hmd_position_global_full_gt_list"])
            self.body_params.append(i["body_parms_list"])
            self.head_motion.append(i["head_global_trans_list"])

        self.motion_54 = []
        self.up_idx = [15, 20, 21]
        self.n_joint = 22
        for idx in range(len(self.sparses)):
            sparse = self.sparses[idx]
            seq = sparse.shape[0]
            # # rot_absolute = self.motions[idx].reshape(-1, self.n_joint, 6)
            # rot_abs = sparse[:, :self.n_joint * 6].reshape(-1, self.n_joint, 6)
            # rot_vel = sparse[:, self.n_joint * 6: self.n_joint * 12].reshape(-1, self.n_joint, 6)
            # pos = sparse[:, self.n_joint * 12: self.n_joint * 15].reshape(-1, self.n_joint, 3)
            # pos_vel = sparse[:, self.n_joint * 15: self.n_joint * 18].reshape(-1, self.n_joint, 3)
            # # sparse_396 = torch.cat((rot_absolute, rot_vel, pos, pos_vel), dim=-1).reshape(seq, -1)  # (seq, 22, 18)
            # sparse_396_abs = torch.cat((rot_abs, rot_vel, pos, pos_vel), dim=-1)
            # self.motion_54.append(sparse_396_abs[:, self.up_idx])
            self.motion_54.append(sparse.reshape(-1, 3, 18))

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx]
        # motion_abs = self.motion_absolute[idx]
        # motion_abs = self.sparses[idx][:, :132]
        sparse = self.motion_54[idx]  # (N, 54)
        body_param = self.body_params[idx]  # {root_orient:(N+1,3), pose_body:(N+1,63), trans:(N+1,3)}
        head_motion = self.head_motion[idx]  # (N, 4, 4)
        filename = self.filename_list[idx]

        return motion, sparse, body_param, head_motion, filename


def get_path_stage1(dataset_path, split):
    assert split in ['test', 'train', 'val']
    subdatasets = ['amass', 'motionx', 'trumans']
    filepaths = []
    if split == 'train':
        rootpath = os.path.join(dataset_path, 'stage1', 'shuffling')
    else:
        rootpath = os.path.join(dataset_path, 'stage1', 'by_order')
    for subds in subdatasets:
        subds_split_dir = os.path.join(rootpath, subds, split)
        filepaths += sorted(glob.glob(os.path.join(subds_split_dir, '*.pt')))
    return filepaths

def get_motion(motion_lists):
    '''
    motion_list: [motion data loaded from motion filepaths]
    '''
    motions = [i['rotation_local_body_gt_list'] for i in motion_lists]  # [(N-1, 22*6)] 身体关节旋转6d表示
    sparses = [i['hmd_position_global_gt_list'] for i in motion_lists]  # [(N-1, 54)]   稀疏控制信号
    return motions, sparses

def get_motion_list_wrapper(motion_lists):
    '''
    motion_list: [motion data loaded from motion filepaths]
    '''
    motion_dict_warpped = []
    for motion_dict in motion_lists:
        d = {}
        d['rotation_local_full_gt_list'] = motion_dict['rotation_local_body_gt_list']
        d['hmd_position_global_full_gt_list'] = motion_dict['hmd_position_global_gt_list']
        d['head_global_trans_list'] = motion_dict['head_global_trans_list']
        origin_body_params = motion_dict['smplx_part_params']  # a dict.
        body_param = {}
        body_param['pose_body'] = origin_body_params['body_pose']
        body_param['trans'] = origin_body_params['transl'].squeeze()
        body_param['root_orient'] = origin_body_params['global_orient'].squeeze()
        d['body_parms_list'] = body_param
        motion_dict_warpped.append(d)
    return motion_dict_warpped


def get_data_stage1(dataset_path, split, protocol, **kwargs):
    motion_list = get_path_stage1(dataset_path, split) # [:512]
    fnamelist = [
        '-'.join((i[:-len(Path(i).suffix)].split('/'))[-4:]) for i in motion_list
    ]
    motion_list = [torch.load(i, map_location='cpu') for i in tqdm(motion_list)]
    assert split in ['test', 'train', 'val']
    if split == 'test':
        motion_list = get_motion_list_wrapper(motion_list)
        return fnamelist, motion_list
    
    assert ("input_motion_length" in kwargs), "Please specify the input_motion_length"
    
    input_motion_length = kwargs["input_motion_length"]
    motions, sparses = get_motion(motion_list)
    new_motions, new_sparses = [], []
    for idx, motion in enumerate(motions):
        if motion.shape[0] < input_motion_length:
            continue
        new_motions.append(motions[idx])
        new_sparses.append(sparses[idx])
    motion_list = get_motion_list_wrapper(motion_list)
    return new_motions, new_sparses, motion_list


def get_path_stage2(dataset_path, split):
    assert split in ['test', 'train', 'val']
    return os.path.join(dataset_path, 'stage2', split, 'normal_motion_data.npy')

def get_motion_list_from_npydata_stage2(motion_npy:dict):
    def np2torch(d:dict):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = np2torch(v)
            elif isinstance(v, np.ndarray):
                d[k] = torch.from_numpy(v)
            elif isinstance(v, list):
                d[k] = torch.tensor(v)
            else:
                d[k] = v
        return d
    motion_list = []
    tensor_keys = list(motion_npy.keys())
    tensor_keys.remove('smplx_part_params')
    body_para_key = 'smplx_part_params'
    body_params = motion_npy[body_para_key]
    l = len(body_params)  # 一共有这么多个片段; 注意其它的要丢掉第1帧
    cnt = 0
    for i in range(l):
        body_param = body_params[i]
        l_seq = body_param['transl'].shape[0]
        motion_info = {
            k: torch.from_numpy(motion_npy[k][cnt+1: cnt+l_seq]) for k in tensor_keys  # 注意去掉一个序列的第一帧
        }
        motion_info[body_para_key] = np2torch(body_param)
        cnt += l_seq
        motion_list.append(motion_info)
    return motion_list

def get_data_stage2(dataset_path, split, protocol, **kwargs):
    npypath = get_path_stage2(dataset_path, split)
    motion_npy = np.load(npypath, allow_pickle=True)
    motion_list = get_motion_list_from_npydata_stage2(motion_npy) # [:512]
    filenames = [f"test: {i}" for i in range(len(motion_list))]
    assert split in ['test', 'train', 'val']
    if split == 'test':
        motion_list = get_motion_list_wrapper(motion_list)
        return filenames, motion_list
    assert ("input_motion_length" in kwargs), "Please specify the input_motion_length"
    
    input_motion_length = kwargs['input_motion_length']
    motions, sparses = get_motion(motion_list)
    new_motions, new_sparses = [], []
    for idx, motion in enumerate(motions):
        if motion.shape[0] < input_motion_length:
            continue
        new_motions.append(motions[idx])
        new_sparses.append(sparses[idx])
    motion_list = get_motion_list_wrapper(motion_list)
    return new_motions, new_sparses, motion_list

def load_data(dataset_path, split, protocol, **kwargs):
    if protocol == 'stage1':
        s1 = get_data_stage1(dataset_path, split, protocol, **kwargs)
        print(f"only load stage 1, {len(s1[0])} sequences.")
        return s1
    elif protocol == 'stage2':
        s2 = get_data_stage2(dataset_path, split, protocol, **kwargs)
        print(f"only load stage 2, {len(s2[0])} sequences.")
        return s2
    elif protocol == 'both':
        s1 = get_data_stage1(dataset_path, split, protocol, **kwargs)
        s2 = get_data_stage2(dataset_path, split, protocol, **kwargs)
        ls1, ls2 = len(s1[0]), len(s2[0])
        for e1, e2 in zip(s1, s2):
            e1.extend(e2)
        print(f"load both stage 1 and 2. len(stage1): {ls1}, len(stage2): {ls2}, total: {len(s1[0])}")
        return s1
    else:
        raise NotImplementedError(f"Invalid protocol str: {protocol}")