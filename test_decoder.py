import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
import numpy as np
import torch
import smplx
from tqdm import tqdm
from collections import Counter

from utils.evaluate import evaluate_prediction, pred_metrics, gt_metrics, all_metrics, metrics_coeffs
from utils.smplBody import BodyModel
from diffusion_stage.parser_util import get_args, merge_file
from VQVAE.transformer_vqvae import TransformerVQVAE
from diffusion_stage.wrap_model import MotionDiffusion
from diffusion_stage.transformer_decoder import TransformerDecoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#####################
lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def overlapping_test_simplify(args, data, dataset, diff_model_upper, diff_model_lower, vq_model_upper, vq_model_lower,
                              decoder_model, num_per_batch=256):
    gt_data, sparse_original, body_param, head_motion, filename = (data[0], data[1], data[2], data[3], data[4])
    sparse_original = sparse_original.cuda().float()  # (seq, 54)
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    # padding the sequence with the first frame
    gt_data_splits = []  # 重叠的序列
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    seq_pad = sparse_original[:1]  # .repeat(args.INPUT_MOTION_LENGTH - 1, 1, 1)
    seq_pad = seq_pad.repeat(args.INPUT_MOTION_LENGTH - 1, 1, 1)
    gt_data_pad = torch.cat((seq_pad, sparse_original), dim=0)  # (31+seq, 54)

    # divide into block
    for i in range(num_frames):
        gt_data_splits.append(gt_data_pad[i: i + block_seq])
    gt_data_splits = torch.stack(gt_data_splits)  # (x, 32, 54)

    # calculate the total batch
    n_steps = gt_data_splits.shape[0] // num_per_batch
    if len(gt_data_splits) % num_per_batch > 0:
        n_steps += 1

    # inference
    output_samples = []
    for step_index in range(n_steps):
        sparse_per_batch = gt_data_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        new_batch_size, new_seq = sparse_per_batch.shape[:2]
        with torch.no_grad():
            upper_latents = diff_model_upper.diffusion_reverse(sparse_per_batch.reshape(new_batch_size, new_seq, 3, 18))
            lower_latents = diff_model_lower.diffusion_reverse(sparse_per_batch.reshape(new_batch_size, new_seq, 3, 18),
                                                               upper_latents)
            recover_6d = decoder_model(upper_latents, lower_latents, sparse_per_batch)
        sample = recover_6d[:, -1].reshape(-1, 22 * 6)
        output_samples.append(sample.cpu().float())
    return output_samples, body_param, head_motion, filename


def test_process(args=None, log_path=None, cur_epoch=None):
    if args is None:
        args = get_args()
        # cfg_args.cfg = 'config_decoder/decoder.yaml'
        # args = merge_file(cfg_args)
        # name = cfg_args.cfg.split('/')[-1].split('.')[0]
        # args.SAVE_DIR = os.path.join("outputs", name)

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
        protocol=args.PROTOCOL
    )
    dataset = TestDataset(all_info, filename_list)

    log = {}
    for metric in all_metrics:
        log[metric] = 0

    # Load VQVAE model
    vqcfg = args.VQVAE
    vq_model_upper = TransformerVQVAE(in_dim=len(upper_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)
    vq_model_lower = TransformerVQVAE(in_dim=len(lower_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)

    # Load Diffusion model
    diff_model_upper = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_upper, use_upper=False).to(device)
    diff_model_lower = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_lower, use_upper=True).to(device)
    decoder_model = TransformerDecoder(in_dim=132, seq_len=args.INPUT_MOTION_LENGTH, **args.DECODER).to(device)

    # Upper VQVAE weight
    upper_vq_dir = args.UPPER_VQ_DIR
    vqvae_upper_file = os.path.join(upper_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_upper_file):
        checkpoint_upper = torch.load(vqvae_upper_file, map_location=lambda storage, loc: storage)
        vq_model_upper.load_state_dict(checkpoint_upper)
        print(f"=> Load upper vqvae {vqvae_upper_file}")
    else:
        print("No upper vqvae model!")
        return

    # Lower VQVAE weight
    lower_vq_dir = args.LOWER_VQ_DIR
    vqvae_lower_file = os.path.join(lower_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_lower_file):
        checkpoint_lower = torch.load(vqvae_lower_file, map_location=lambda storage, loc: storage)
        vq_model_lower.load_state_dict(checkpoint_lower)
        print(f"=> Load upper vqvae {vqvae_lower_file}")
    else:
        print("No lower vqvae model!")
        return

    output_dir = args.SAVE_DIR
    output_file = os.path.join(output_dir, 'best.pth.tar')
    if os.path.exists(output_file):
        checkpoint_all = torch.load(output_file, map_location=lambda storage, loc: storage)
        diff_model_upper.load_state_dict(checkpoint_all['upper_state_dict'])
        diff_model_lower.load_state_dict(checkpoint_all['lower_state_dict'])
        decoder_model.load_state_dict(checkpoint_all['decoder_state_dict'])
        print("=> loading checkpoint '{}'".format(output_file))
    else:
        print("backbone not exist")

    vq_model_upper.eval()
    vq_model_lower.eval()
    diff_model_upper.eval()
    diff_model_lower.eval()
    decoder_model.eval()

    ########## From XRZ ###########
    smplx_model = smplx.create(
        model_path = args.SMPLX_DIR,
        model_type='smplx',
        gender='neutral',
        ext='npz'
    )
    ########## From XRZ ###########
    
    # 下面这么写是为了tqdm的兼容性,需要可视化时显示每个视频的可视化进度,光测试不可视化时显示测试进度
    if args.VIS or args.BVH:
        test_loader = range(len(dataset))
    else:
        test_loader = tqdm(range(len(dataset)))

    for sample_index in test_loader:
        output, body_param, head_motion, filename = (
            overlapping_test_simplify(args, dataset[sample_index], dataset, diff_model_upper, diff_model_lower,
                                      vq_model_upper, vq_model_lower, decoder_model, 1024))
        sample = torch.cat(output, dim=0)  # (N, 132) N表示帧数

        instance_log = evaluate_prediction(
            args, all_metrics, sample, body_model,
            head_motion, body_param, fps, filename, smplx_model     # From XRZ: Add smplx_model parameter
        )
        for key in instance_log:
            log[key] += instance_log[key]

    print("Metrics for the predictions")
    result_str = "\n"
    if cur_epoch is not None:
        result_str += f"epoch{cur_epoch} \n"
    for metric in pred_metrics:
        result_str += f"{metric}: {log[metric] / len(dataset) * metrics_coeffs[metric]} \n"
    for metric in gt_metrics:
        result_str += f"{metric}: {log[metric] / len(dataset) * metrics_coeffs[metric]} \n"
    print(result_str)
    if log_path is not None:
        with open(log_path, 'a') as f:
            f.write(result_str)
            print(f"Evalution results save to {log_path}")

    return log["mpjpe"] / len(dataset) * metrics_coeffs["mpjpe"]


if __name__ == "__main__":
    test_process()
