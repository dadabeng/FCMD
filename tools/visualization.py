import os
import sys
import argparse
import codecs as cs
import torch
import numpy as np
from os.path import join as pjoin

# Make the repository root importable so that the script can be run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.paramUtil as paramUtil
from utils.plot_script import plot_3d_motion_ST2M
from utils.get_opt import get_opt
from models import T2MUnet
from models import DiffusePipeline
from utils.utils import motion_temporal_filter
from utils.motion_process import recover_from_ric
from utils.model_load import load_model_weights


def slerp_translation(last_transl, new_transl, number_of_frames):
    alpha = torch.linspace(0, 1, number_of_frames + 2)
    # 2 more than needed
    inter_trans = torch.einsum("i,...->i...", 1 - alpha, last_transl) + torch.einsum("i,...->i...", alpha, new_transl)
    return inter_trans[1:-1]


def do_slerp_op(motion, lengths, slerp_window_size=4):
    pose = motion.clone()
    end_first_motion = lengths[0] - 1
    for length in lengths[1:]:
        begin_second_motion = end_first_motion + 1
        begin_second_motion += slerp_window_size

        inter_pose = slerp_translation(pose[end_first_motion], pose[begin_second_motion], slerp_window_size)
        pose[end_first_motion + 1:begin_second_motion] = inter_pose

        end_first_motion += length
    return pose


def plot_t2m(data, motion_length, result_path, npy_path, caption, joints_num):
    joints = recover_from_ric(torch.from_numpy(data).float(), joints_num).numpy()
    joints = motion_temporal_filter(joints, sigma=1)
    plot_3d_motion_ST2M(save_path=result_path,
                        motion_length=motion_length,
                        kinematic_tree=paramUtil.t2m_kinematic_chain,
                        joints=joints,
                        title=caption,
                        fps=20)
    if npy_path is not None:
        np.save(npy_path, joints)


def build_models(opt):
    print('\nInitializing model ...')
    model = T2MUnet(
        input_feats=opt.dim_pose,
        text_latent_dim=opt.text_latent_dim,
        base_dim=opt.base_dim,
        dim_mults=opt.dim_mults,
        time_dim=opt.time_dim,
        adagn=not opt.no_adagn,
        zero=True,
        no_eff=opt.no_eff,
        cond_mask_prob=getattr(opt, 'cond_mask_prob', 0.)
    )
    return model


def read_grained_text(file_path, num_segments):
    """Read the fine-grained text file: 6 short phrases per segment
    (head / torso / left arm / right arm / left leg / right leg)."""
    grained_text_data = []
    grained_text = []
    with cs.open(file_path) as f:
        num = 0
        for line in f.readlines():
            num += 1
            line = line.strip("\n")
            grained_text.append(line)
            if num == 6:
                grained_text_data.append(grained_text)
                grained_text = []
                num = 0
    assert len(grained_text_data) == num_segments, \
        'the grained text file should contain 6 lines per segment, got %d groups for %d segments' % (
            len(grained_text_data), num_segments)
    return grained_text_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, required=True,
                        help='Path to the opt.txt of the trained model')
    parser.add_argument('--text', type=str, nargs='+', required=True,
                        help='One caption per motion segment, '
                             'e.g. --text "sit down" "raise the right hand" "crawl"')
    parser.add_argument('--motion_length', type=int, nargs='+', required=True,
                        help='Expected length (frames) of each segment, e.g. --motion_length 60 40 60')
    parser.add_argument('--grained_text_file', type=str, required=True,
                        help='Fine-grained text file, 6 lines per segment, '
                             'see examples/grained_text_example.txt')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Path to save the generated gif (default: result/<model name>/teaser.gif)')
    parser.add_argument('--npy_path', type=str, default=None,
                        help='Optional, save the joint coordinates of each frame, shape (T, 22, 3)')
    parser.add_argument('--gpu_id', type=int, default=0, help="Which gpu to use, -1 for cpu")
    parser.add_argument('--is_slerp', action='store_true',
                        help='Interpolate a 4-frame gap between segments with slerp')
    parser.add_argument('--which_epoch', type=str, default='model_50000',
                        help='Checkpoint to load, e.g. model_50000 or latest')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                        help='Number of denoising steps')
    parser.add_argument('--diffuser_name', type=str, default='dpmsolver',
                        help="Sampler's scheduler name (see config/diffuser_params.yaml)")
    args = parser.parse_args()

    if args.gpu_id != -1 and torch.cuda.is_available():
        device = torch.device('cuda:%d' % args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device('cpu')

    opt = get_opt(args.opt_path, device)
    opt.no_ema = False
    opt.num_inference_steps = args.num_inference_steps
    opt.diffuser_name = args.diffuser_name
    opt.which_epoch = args.which_epoch

    metas = args.text
    motion_length = args.motion_length
    assert len(metas) == len(motion_length), 'number of captions and motion lengths must match'
    max_length = max(motion_length)
    assert max_length >= 16 and max_length <= 196, 'motion length should be in [16, 196]'

    grained_text_data = read_grained_text(args.grained_text_file, len(motion_length))

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    encoder = build_models(opt).to(device)
    ckpt_path = pjoin(opt.model_dir, opt.which_epoch + '.tar')
    it = load_model_weights(encoder, ckpt_path, use_ema=not opt.no_ema)

    if args.result_path is None:
        directory_path = pjoin('./result', opt.name)
        result_path = pjoin(directory_path, 'teaser.gif')
    else:
        directory_path = os.path.dirname(args.result_path)
        result_path = args.result_path
    if directory_path:
        os.makedirs(directory_path, exist_ok=True)

    pipeline = DiffusePipeline(
        opt=opt,
        encoder=encoder,
        diffuser_name=opt.diffuser_name,
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )

    num_intervals = len(motion_length)
    is_slerp = args.is_slerp
    with torch.no_grad():
        lens = torch.tensor(motion_length)
        m_lens = torch.tensor(motion_length)
        if is_slerp:
            for i in range(num_intervals):
                if i != 0:
                    m_lens[i] = m_lens[i] - 4
        m_lens = torch.unsqueeze(m_lens, 1).to(device)
        pred_motions = pipeline.generate_ST2M(metas, m_lens, grained_text_data, opt.dim_pose)
        all_pred_motion = []
        for i in range(num_intervals):
            motion = pred_motions[i][0]
            motion = motion[:int(m_lens[i][0])].cpu()
            if is_slerp:
                if i != 0:
                    slerp_gap_motion = torch.zeros([4, 263], dtype=torch.float64)
                    motion = torch.cat((slerp_gap_motion, motion), 0)
            all_pred_motion.append(motion)
        pred_motion = all_pred_motion[0]
        for i in range(1, num_intervals):
            pred_motion = torch.cat((pred_motion, all_pred_motion[i]), dim=0)
        if is_slerp:
            pred_motion = do_slerp_op(pred_motion, lens)
        pred_motion = pred_motion.numpy()
        pred_motion = pred_motion * std + mean
        if is_slerp:
            for i in range(num_intervals):
                if i != 0:
                    m_lens[i] = m_lens[i] + 4
        plot_t2m(pred_motion, m_lens, result_path, args.npy_path, metas, opt.joints_num)
