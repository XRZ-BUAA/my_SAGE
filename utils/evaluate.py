import torch
import os
import math
import smplx
from utils import utils_visualize
# from utils import utils_transform
from utils.transform_tools import rotation_6d_to_axis_angle
from utils.metrics import get_metric_function
from utils.utils_bvh import export_bvh

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0
pred_metrics = [
    "mpjre",
    "upperre",
    "lowerre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "upperre": RADIANS_TO_DEGREES,
    "lowerre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
    "gt_mpjpe": METERS_TO_CENTIMETERS,
    "gt_mpjve": METERS_TO_CENTIMETERS,
    "gt_handpe": METERS_TO_CENTIMETERS,
    "gt_rootpe": METERS_TO_CENTIMETERS,
    "gt_upperpe": METERS_TO_CENTIMETERS,
    "gt_lowerpe": METERS_TO_CENTIMETERS,
}
# upper/lower_index are used to evaluate the results following AGRoL
upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
# upper_body_part is not the same as upper_index
upper_body_part = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_body_part = [0, 1, 2, 4, 5, 7, 8, 10, 11]

# SMPLX 的关节
TORSO_JOINT_NAMES = [
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
    # "jaw",
    # "left_eye_smplhf",
    # "right_eye_smplhf",
    # "left_index1",  # 从这里开始是左手 25
    # "left_index2",
    # "left_index3",
    # "left_middle1",
    # "left_middle2",
    # "left_middle3",
    # "left_pinky1",
    # "left_pinky2",
    # "left_pinky3",
    # "left_ring1",
    # "left_ring2",
    # "left_ring3",
    # "left_thumb1",
    # "left_thumb2",
    # "left_thumb3",  # 左手结束 39
    # "right_index1", # 右手开始 40
    # "right_index2",
    # "right_index3",
    # "right_middle1",
    # "right_middle2",
    # "right_middle3",
    # "right_pinky1",
    # "right_pinky2",
    # "right_pinky3",
    # "right_ring1",
    # "right_ring2",
    # "right_ring3",
    # "right_thumb1",
    # "right_thumb2",
    # "right_thumb3", # 右手结束 54
]


def evaluate_prediction(args, metrics, sample, body_model, head_motion, body_param, fps, filename, smplx_model,     # From XRZ: Add smplx_model as input
                        use_body_part="full"):
    motion_pred = sample.squeeze().cuda()  # (seq_len, 132)
    seq_len = motion_pred.shape[0]
    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-seq_len:, ...]

    # Get the  prediction from the model
    model_rot_input = (  # (N, 66)
        rotation_6d_to_axis_angle(motion_pred.reshape(-1, 6).detach()).reshape(motion_pred.shape[0], -1).float()
    )
    assert use_body_part in ["upper", "lower", "full"]
    if use_body_part == "upper":
        pred_full_tmp = torch.zeros((seq_len, 22, 3)).to(model_rot_input.device)
        # pred_full_tmp[:, upper_body_part] = model_rot_input.reshape(seq_len, len(upper_body_part), 3)
        pred_full_tmp[:, upper_body_part] = model_rot_input.reshape(seq_len, -1, 3)[:, upper_body_part]
        model_rot_input = pred_full_tmp.reshape(seq_len, 66)
        body_param['pose_body'].reshape(seq_len, 21, 3)[:, [0, 1, 3, 4, 6, 7, 9, 10]] *= 0.0
    elif use_body_part == "lower":
        pred_full_tmp = torch.zeros((seq_len, 22, 3)).to(model_rot_input.device)
        # pred_full_tmp[:, lower_body_part] = model_rot_input.reshape(seq_len, len(lower_body_part), 3)
        pred_full_tmp[:, lower_body_part] = model_rot_input.reshape(seq_len, -1, 3)[:, lower_body_part]
        model_rot_input = pred_full_tmp.reshape(seq_len, 66)
        body_param['pose_body'].reshape(seq_len, 21, 3)[:, [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]] *= 0.0

    T_head2world = head_motion.clone().cuda()
    t_head2world = T_head2world[:, :3, 3].clone()
    # Get the offset between the head and other joints using forward kinematic model
    # body_pose_local: (batch, 52, 3) joints location
    body_pose_local = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]  # root - head location
    t_root2world = t_head2root + t_head2world.cuda()

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            # "trans": body_param['trans'],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    # Get the  ground truth position from the model
    gt_body = body_model(body_param)
    gt_position = gt_body.Jtr[:, :22, :]

    # Create animation
    # "CMU-94", "CMU-55", "CMU-14", "CMU-206", "MPI_HDM05-20", "BioMotionLab_NTroje-26"
    if args.VIS:
        video_dir = os.path.join(args.VIS_DIR, "Video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        save_filename = filename.split(".")[0].replace("/", "-")
        save_video_path = os.path.join(video_dir, save_filename + ".mp4")
        utils_visualize.save_animation(
            body_pose=predicted_body,
            savepath=save_video_path,
            bm=body_model.body_model,
            fps=fps,
            resolution=(800, 800),
        )
        if args.SAVE_GT:
            save_video_path_gt = os.path.join(video_dir, save_filename + "_gt.mp4")
            if not os.path.exists(save_video_path_gt):
                utils_visualize.save_animation(
                    body_pose=gt_body,
                    savepath=save_video_path_gt,
                    bm=body_model.body_model,
                    fps=fps,
                    resolution=(800, 800),

                )

    if args.BVH:
        save_filename = filename.split(".")[0].replace("/", "-")
        bvh_dir = os.path.join(args.VIS_DIR, "BVH")
        if not os.path.exists(bvh_dir):
            os.makedirs(bvh_dir)
        save_bvh_path = os.path.join(bvh_dir, save_filename + ".bvh")
        export_bvh(
            smplx_model,
            model_rot_input.reshape(seq_len, -1, 3),
            t_root2world,
            save_bvh_path,
            jnames=TORSO_JOINT_NAMES,
            fps=fps
        )
        if args.BVH_GT:
            save_bvh_path_gt = os.path.join(bvh_dir, save_filename + "_gt.bvh")
            export_bvh(
                smplx_model,
                torch.cat([body_param["root_orient"], body_param["pose_body"]], dim=-1).reshape(seq_len, -1, 3),
                body_param["trans"],
                save_bvh_path_gt,
                jnames=TORSO_JOINT_NAMES,
                fps=fps
            )
            # raise NotImplementedError
    # import pickle
    # video_dir = "SAGENet/outputs/plot_result"
    # if not os.path.exists(video_dir):
    #     os.makedirs(video_dir)
    #
    # save_filename = filename.split(".")[0].replace("/", "-")
    # save_file_path_gt = os.path.join(video_dir, save_filename + "_SAGENET.pkl")
    # posi = predicted_body.Jtr.cpu().numpy()
    # verts = predicted_body.v.cpu().numpy()
    # faces = predicted_body.f.cpu().numpy()
    # res = {
    #     "pos": posi,
    #     "verts": verts,
    #     "face": faces
    # }
    # file = open(save_file_path_gt, 'wb')
    # print(f"saving {save_file_path_gt}")
    # pickle.dump(res, file)
    # file.close()

    gt_angle = body_param["pose_body"]
    gt_root_angle = body_param["root_orient"]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                predicted_angle,
                predicted_root_angle,
                gt_position,
                gt_angle,
                gt_root_angle,
                upper_index,
                lower_index,
                fps,
            ).cpu().numpy()
        )

    torch.cuda.empty_cache()
    return eval_log
