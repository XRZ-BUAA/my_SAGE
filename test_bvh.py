import torch
import smplx

from utils.utils_bvh import export_bvh
from utils.transform_tools import rotation_6d_to_axis_angle
from utils import utils_transform
# SMPLX 的关节
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",  # 这到这前面都是传统身体关节，21
]

body_model = smplx.create(
    model_path='body_models',
    model_type='smplx',
    gender='neutral',
    ext='npz',
)

file = '../2508.pt'
data = torch.load(file)
params = data['smplx_part_params']
transl = params['transl'][1:]

local_rot = data['rotation_local_body_gt_list']
# print(local_rot.shape)
nseq = local_rot.shape[0]
rot_aa = utils_transform.sixd2aa(local_rot.reshape(-1, 6)).reshape(nseq, -1, 3)
# print(rot_aa.shape)

# full_rot = torch.cat(
#     [
#         params['global_orient'].reshape(nseq, -1, 3),
#         params['body_pose'].reshape(nseq, -1, 3),
#         torch.ones(nseq, 3, 3),
#         params['left_hand_pose'].reshape(nseq, -1, 3),
#         params['right_hand_pose'].reshape(nseq, -1, 3),
        
#     ],
#     dim=-2
# )
# print(full_rot[100])
# export_bvh(JOINT_NAMES, body_model, full_rot, transl, 'test.bvh')
export_bvh(
    body_model,
    rot_aa,
    transl,
    'test_local.bvh',
    jnames=JOINT_NAMES,
    fps=30
    
)