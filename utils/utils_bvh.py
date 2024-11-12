import os
import re
import smplx
import numpy as np

from utils import quat

if os.name == 'nt':
    from pytorch3d.pytorch3d.transforms.rotation_conversions import axis_angle_to_quaternion
else:
    from pytorch3d.transforms.rotation_conversions import axis_angle_to_quaternion


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
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",  # 从这里开始是左手 25
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",  # 左手结束 39
    "right_index1", # 右手开始 40
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3", # 右手结束 54
#     "nose",
#     "right_eye",
#     "left_eye",
#     "right_ear",
#     "left_ear",
#     "left_big_toe",
#     "left_small_toe",
#     "left_heel",
#     "right_big_toe",
#     "right_small_toe",
#     "right_heel",
#     "left_thumb",
#     "left_index",
#     "left_middle",
#     "left_ring",
#     "left_pinky",
#     "right_thumb",
#     "right_index",
#     "right_middle",
#     "right_ring",
#     "right_pinky",
#     "right_eye_brow1",
#     "right_eye_brow2",
#     "right_eye_brow3",
#     "right_eye_brow4",
#     "right_eye_brow5",
#     "left_eye_brow5",
#     "left_eye_brow4",
#     "left_eye_brow3",
#     "left_eye_brow2",
#     "left_eye_brow1",
#     "nose1",
#     "nose2",
#     "nose3",
#     "nose4",
#     "right_nose_2",
#     "right_nose_1",
#     "nose_middle",
#     "left_nose_1",
#     "left_nose_2",
#     "right_eye1",
#     "right_eye2",
#     "right_eye3",
#     "right_eye4",
#     "right_eye5",
#     "right_eye6",
#     "left_eye4",
#     "left_eye3",
#     "left_eye2",
#     "left_eye1",
#     "left_eye6",
#     "left_eye5",
#     "right_mouth_1",
#     "right_mouth_2",
#     "right_mouth_3",
#     "mouth_top",
#     "left_mouth_3",
#     "left_mouth_2",
#     "left_mouth_1",
#     "left_mouth_5",  # 59 in OpenPose output
#     "left_mouth_4",  # 58 in OpenPose output
#     "mouth_bottom",
#     "right_mouth_4",
#     "right_mouth_5",
#     "right_lip_1",
#     "right_lip_2",
#     "lip_top",
#     "left_lip_2",
#     "left_lip_1",
#     "left_lip_3",
#     "lip_bottom",
#     "right_lip_3",
#     # Face contour
#     "right_contour_1",
#     "right_contour_2",
#     "right_contour_3",
#     "right_contour_4",
#     "right_contour_5",
#     "right_contour_6",
#     "right_contour_7",
#     "right_contour_8",
#     "contour_middle",
#     "left_contour_8",
#     "left_contour_7",
#     "left_contour_6",
#     "left_contour_5",
#     "left_contour_4",
#     "left_contour_3",
#     "left_contour_2",
#     "left_contour_1",
]


channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}



def save_joint(f, data, t, i, save_order, order='zyx', save_positions=False):
    
    save_order.append(i)
    
    f.write("%sJOINT %s\n" % (t, data['names'][i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, data['offsets'][i,0], data['offsets'][i,1], data['offsets'][i,2]))
    
    if save_positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(len(data['parents'])):
        if data['parents'][j] == i:
            t = save_joint(f, data, t, j, save_order, order=order, save_positions=save_positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t
    

def save(filename, data, save_positions=False):
    """ Save a joint hierarchy to a file.
    
    Args:
        filename (str): The output will save on the bvh file.
        data (dict): The data to save.(rotations, positions, offsets, parents, names, order, frametime)
        save_positions (bool): Whether to save all of joint positions on MOTION. (False is recommended.)
    """
    
    order = data['order']
    frametime = data['frametime']
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, data['names'][0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, data['offsets'][0,0], data['offsets'][0,1], data['offsets'][0,2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        save_order = [0]
            
        for i in range(len(data['parents'])):
            if data['parents'][i] == 0:
                t = save_joint(f, data, t, i, save_order, order=order, save_positions=save_positions)
    
        t = t[:-1]
        f.write("%s}\n" % t)

        rots, poss = data['rotations'], data['positions']

        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots));
        f.write("Frame Time: %f\n" % frametime);
        
        for i in range(rots.shape[0]):
            for j in save_order:
                
                if save_positions or j == 0:
                
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0], poss[i,j,1], poss[i,j,2], 
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                
                else:
                    
                    f.write("%f %f %f " % (
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))

            f.write("\n")


def export_bvh(
        body_model,
        rots,
        transl,
        save_file,
        jnames=JOINT_NAMES,
        fps=30, 
):
    '''
    kin_table: 
    jnames: 关节名列表
    body_model: 用于计算relative joints.
    rots: (n_seq, n_joints, 3), 轴角.  
    transl: (n_seq, 3), 平移
    '''
    parents = body_model.parents.detach().cpu().numpy()[:len(jnames)]
    # print(parents)
    rest = body_model()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:len(jnames), :]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100      # ?

    rots = axis_angle_to_quaternion(rots).cpu().numpy()
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:, 0] += transl.cpu().numpy() * 100     # ?
    rotations = np.degrees(quat.to_euler(rots, order=order))

    bvh_data = {
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": jnames,
        "order": order,
        "frametime": 1 / fps,
    }

    if not save_file.endswith('.bvh'):
        save_file += '.bvh'
    save(save_file, bvh_data)


    
