import os
import math
import random
import numpy as np
import torch
from tqdm import tqdm
from utils import utils_transform
from utils.metrics import get_metric_function

import smplx
# from utils.evaluate import evaluate_prediction
from utils.utils_bvh import export_bvh
from utils.evaluate import TORSO_JOINT_NAMES

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from VQVAE.parser_util import get_args
from VQVAE.transformer_vqvae import TransformerVQVAE
from utils.smplBody import BodyModel

lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
device = "cuda" if torch.cuda.is_available() else "cpu"

# upper/lower_index are used to evaluate the results following AGRoL
upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
# upper_body_part is not the same as upper_index
upper_body_part = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_body_part = [0, 1, 2, 4, 5, 7, 8, 10, 11]

#####################
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "upperre",
    "lowerre",
    "rootre",
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
    "rootre": RADIANS_TO_DEGREES,
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


def overlapping_test_simplify(args, data, models, num_per_batch=256):
    gt_data, sparse_original, body_param, head_motion, filename = (data[0], data[1], data[2], data[3], data[4])
    num_frames = head_motion.shape[0]
    gt_data = gt_data.cuda().float()  # (seq, 132)
    sparse = sparse_original.cuda().float().reshape(num_frames, 54)
    head_motion = head_motion.cuda().float()

    gt_data_splits = []
    sparse_splits = []
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    seq_pad = gt_data[:1].repeat(block_seq - 1, 1)
    sparse_pad = sparse[:1].repeat(block_seq - 1, 1)
    gt_data_pad = torch.cat((seq_pad, gt_data), dim=0)  # (31+seq, 396)
    sparse_pad = torch.cat((sparse_pad, sparse), dim=0)

    for i in range(num_frames):
        gt_data_splits.append(gt_data_pad[i: i + block_seq])
        sparse_splits.append(sparse_pad[i: i + block_seq])

    gt_data_splits = torch.stack(gt_data_splits)  # (x, 32, 396)
    sparse_splits = torch.stack(sparse_splits)

    n_steps = gt_data_splits.shape[0] // num_per_batch
    if len(gt_data_splits) % num_per_batch > 0:
        n_steps += 1

    upper_model = models["upper"]
    lower_model = models["lower"]
    output_samples = []
    output_gts = []

    for step_index in range(n_steps):
        out_gt = None
        gt_per_batch = gt_data_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        sparse_per_batch = sparse_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        sample = torch.zeros_like(gt_per_batch, device=device)
        with torch.no_grad():
            bs, seq = gt_per_batch.shape[:2]
            gt_per_batch = gt_per_batch.reshape((bs, seq, -1, 6))
            sample = sample.reshape((bs, seq, -1, 6))
            # gt_per_batch = gt_per_batch[:, :, body_part, :].reshape((bs, seq, -1))
            upper_gt_per_batch = gt_per_batch[:, :, upper_body, :]
            lower_gt_per_batch = gt_per_batch[:, :, lower_body, :]
            upper_sample, _, indices = upper_model(x=upper_gt_per_batch, sparse=sparse_per_batch)
            lower_sample, _, indices = lower_model(x=lower_gt_per_batch, sparse=sparse_per_batch)
            # sample, _, indices = model(x=gt_per_batch, sparse=sparse_per_batch)
        sample[:, :, upper_body, :] = upper_sample.reshape((bs, seq, -1, 6))
        sample[:, :, lower_body, :] = lower_sample.reshape((bs, seq, -1, 6))
        # sample = sample.reshape((bs, seq, -1))
        sample = sample[:, -1].reshape(-1, 22 * 6)
        out_gt = gt_per_batch[:, -1].reshape(-1, 22 * 6)
        # sample = utils_transform.absSixd2rel_pavis_seq(sample)  # (seq, 132)
        output_samples.append(sample.cpu().float())
        output_gts.append(out_gt.cpu().float())
    # gt_data2 = utils_transform.absSixd2rel_pavis_seq(gt_data[0])
    return output_samples, body_param, head_motion, filename, output_gts    


def evaluate_prediction(args, metrics, sample, gt_data, body_model, head_motion, body_param, fps, filename, smplx_model,     # From XRZ: Add smplx_model as input
                        use_body_part="full"):
    motion_pred = sample.squeeze().cuda()  # (seq_len, 132)
    gt_motion = gt_data.squeeze().cuda()  # (seq_len, 132)
    seq_len = motion_pred.shape[0]
    for k, v in body_param.items():
        body_param[k] = v.squeeze().cuda()
        body_param[k] = body_param[k][-seq_len:, ...]

    # Get the  prediction from the model
    model_rot_input = (  # (N, 66)
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach()).reshape(motion_pred.shape[0], -1).float()
    )
    gt_motion_aa = (
        utils_transform.sixd2aa(gt_motion.reshape(-1, 6).detach()).reshape(gt_motion.shape[0], -1).float()
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
    # gt_body = body_model(body_param)
    gt_body = body_model(
        {
            "pose_body": gt_motion_aa[..., 3:66],
            "root_orient": gt_motion_aa[..., :3],
            "trans": body_param['trans'],
        }
    )
    gt_position = gt_body.Jtr[:, :22, :]

    # Create animation
    # "CMU-94", "CMU-55", "CMU-14", "CMU-206", "MPI_HDM05-20", "BioMotionLab_NTroje-26"

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
                # torch.cat([body_param["root_orient"], body_param["pose_body"]], dim=-1).reshape(seq_len, -1, 3),
                gt_motion_aa.reshape(seq_len, -1, 3),
                body_param["trans"],
                save_bvh_path_gt,
                jnames=TORSO_JOINT_NAMES,
                fps=fps
            )


    # gt_angle = body_param["pose_body"]
    # gt_root_angle = body_param["root_orient"]
    gt_angle = gt_motion_aa[..., 3:66]
    gt_root_angle = gt_motion_aa[..., :3]

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


def test_process():
    args = get_args()
    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    print("USE OURS: ", hasattr(args, 'USE_OURS') and args.USE_OURS)

    fps = args.FPS  # AMASS dataset requires 60 frames per second
    body_model = BodyModel(args.SUPPORT_DIR, smplx=hasattr(args, 'USE_OURS') and args.USE_OURS).to(device)
    print("Loading dataset...")

    if hasattr(args, 'USE_OURS') and args.USE_OURS:
        from dataloader.dataloader_our_wrapper import load_data, TestDataset
    else:
        from dataloader.dataloader import load_data, TestDataset

    filename_list, all_info = load_data(
        args.DATASET_PATH,
        "test",
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
    )
    dataset = TestDataset(all_info, filename_list)

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    vqcfg = args.VQVAE
    upper_model = TransformerVQVAE(in_dim=len(upper_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim, heads=vqcfg.heads,
                             dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook, n_e=vqcfg.n_e, e_dim=vqcfg.e_dim,
                             beta=vqcfg.beta).to(device).eval()
    lower_model = TransformerVQVAE(in_dim=len(lower_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim, heads=vqcfg.heads,
                             dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook, n_e=vqcfg.n_e, e_dim=vqcfg.e_dim,
                             beta=vqcfg.beta).to(device).eval()
    
    upper_file = os.path.join(args.UPPER_DIR, 'best.pth.tar')
    lower_file = os.path.join(args.LOWER_DIR, 'best.pth.tar')
    if os.path.exists(upper_file):
        print("=> loading upper model '{}'".format(upper_file))
        checkpoint = torch.load(upper_file, map_location=lambda storage, loc: storage)
        upper_model.load_state_dict(checkpoint)
    else:
        print(f"{upper_file} not exist!!!")
        return
    if os.path.exists(lower_file):
        print("=> loading lower model '{}'".format(lower_file))
        checkpoint = torch.load(lower_file, map_location=lambda storage, loc: storage)
        lower_model.load_state_dict(checkpoint)
    else:
        print(f"{lower_file} not exist!!!")
        return
    
    models = {
        'upper': upper_model,
        'lower': lower_model
    }
    smplx_model = smplx.create(
        model_path = args.SMPLX_DIR,
        model_type='smplx',
        gender='neutral',
        ext='npz'
    )

    n_testframe = args.NUM_PER_BATCH
    for sample_index in tqdm(range(len(dataset))):
        output, body_param, head_motion, filename, gts = \
            overlapping_test_simplify(args, dataset[sample_index], models, n_testframe)
        sample = torch.cat(output, dim=0)  # (N, 132) N表示帧数
        gt_eval = torch.cat(gts, dim=0)  # (N, 132) N表示帧数
        instance_log = evaluate_prediction(
            args, all_metrics, sample, gt_eval, body_model, head_motion,
            body_param, fps, filename, smplx_model)
        for key in instance_log:
            log[key] += instance_log[key]
    # Print the value for all the metrics
    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])
    print("Metrics for the ground truth")
    for metric in gt_metrics:
        print(metric, log[metric] / len(dataset) * metrics_coeffs[metric])


if __name__ == "__main__":
    test_process()
