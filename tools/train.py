import os
import sys
from os.path import join as pjoin

# Make the repository root importable so that the script can be run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from options.train_options import TrainOptions
from models import T2MUnet
from trainers import DDPMTrainer
from datasets import st2m_Text2Motion_withpast_DatasetV5
from utils.ema import ExponentialMovingAverage

from accelerate.utils import set_seed
from accelerate import Accelerator


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


if __name__ == '__main__':
    accelerator = Accelerator()
    parser = TrainOptions()
    opt = parser.parse(accelerator)
    set_seed(opt.seed)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    opt.no_ema = False

    if accelerator.is_main_process:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)

    if opt.dataset_name == 'BABEL_TEACH':
        opt.data_root = './data/BABEL_TEACH'
        opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.grained_text_dir = pjoin(opt.data_root, 'the fine-grained text')
        opt.joints_num = 22
        opt.max_motion_length = 196
        opt.dim_pose = 263
        train_split_file = pjoin(opt.data_root, 'train_12103.txt')
    elif opt.dataset_name == 'STDM':
        opt.data_root = './data/STDM'
        opt.motion_dir = pjoin(opt.data_root, 'STDM_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'STDM_texts_5289')
        opt.joints_num = 22
        opt.max_motion_length = 196
        opt.dim_pose = 263
        train_split_file = pjoin(opt.data_root, 'train_4231.txt')
    else:
        raise KeyError('Dataset Does NOT Exist')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    accelerator.print('\nInitializing model ...')
    encoder = build_models(opt)
    model_ema = None
    if opt.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        adjust = 106_667 * opt.model_ema_steps / opt.num_train_steps
        alpha = 1.0 - opt.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        print('EMA alpha:', alpha)
        model_ema = ExponentialMovingAverage(encoder, decay=1.0 - alpha)
    accelerator.print('Finish building Model.\n')

    trainer = DDPMTrainer(opt, encoder, accelerator, model_ema)
    train_dataset = st2m_Text2Motion_withpast_DatasetV5(opt, mean, std, train_split_file, accelerator=accelerator)

    trainer.train(train_dataset)
