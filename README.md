# FCMD: Text-Driven Human Motion Generation

This is the official PyTorch implementation of
**FCMD: Fine-Grained Text-Driven Cohesive Motion Generation With Diffusion Model**
([IEEE Xplore](https://ieeexplore.ieee.org/document/11456743)).

<p align="center">
  <img src="figures/teaser.png" width="100%">
</p>
<p align="center">
  <i>A generated motion sequence ("walk" &rarr; "jump" &rarr; "crawl"). Each segment is driven by a coarse caption together with fine-grained per-body-part descriptions (head / torso / left arm / right arm / left leg / right leg).</i>
</p>

Given a sequence of motion segments described by coarse captions together with fine-grained
per-body-part text descriptions, FCMD generates a long, coherent human motion sequence in an
autoregressive manner: a diffusion-based 1D UNet denoiser is conditioned on CLIP text features
fused with fine-grained text features (FiLM), and every newly generated segment is conditioned on
the previously generated motion to keep the whole sequence temporally consistent.

## Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference / Visualization](#inference--visualization)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Installation

Requirements: Linux, Python 3.9, CUDA 11.3 (any compatible PyTorch/CUDA pair should work).

```shell
conda create -n fcmd python=3.9 -y
conda activate fcmd

# PyTorch (matching your CUDA version)
pip install torch==1.12.1 torchvision==0.13.1 -f https://download.pytorch.org/whl/cu113

# Other dependencies
pip install -r requirements.txt
```

As an alternative, the exact environment used for our experiments is provided in
[`environment.yml`](environment.yml):

```shell
conda env create -f environment.yml
conda activate mogen
```

`ffmpeg` is required to export the visualization gifs (install it via conda/apt if missing).

## Data Preparation

### BABEL_TEACH dataset

Download the **BABEL_TEACH** dataset from the [ST2M release](https://github.com/Druthrie/ST2M)
([Google Drive](https://drive.google.com/file/d/1_KCzpH6BA-7BnL2_QrkJa2CA_RQUAvWR/view?usp=sharing))
and place it under `data/BABEL_TEACH`:

```text
data/BABEL_TEACH/
├── Mean.npy / Std.npy                      # per-feature normalization statistics
├── train_12103.txt / val_2163.txt / test_2000.txt
├── BABEL_TEACH_joint_vecs_2/               # motion features, shape (frames, 263)
│   └── <id>_C000.npy, <id>_C001.npy        # two consecutive segments per sample
├── BABEL_TEACH_texts/                      # coarse captions, one line per segment
│   └── <id>.txt                            # format: "caption#word/POS word/POS ..."
└── the fine-grained text/                  # fine-grained descriptions, 6 lines per segment
    └── <id>.txt                            # head / torso / left arm / right arm / left leg / right leg
```

The code also supports the **STDM** dataset with the same layout (`data/STDM/`, see
`tools/train.py` for the expected file names); it is available from the same ST2M release
([Google Drive](https://drive.google.com/file/d/1q6PgN2Nut7fuAlEXZITA7gBMDehUrkJC/view?usp=sharing),
request procedure in the ST2M README).

### Evaluation models

Matching score, R-precision and FID are computed with the text–motion contrastive models of
[text-to-motion](https://github.com/EricGuo5513/text-to-motion). Download the pretrained
contrastive model for BABEL_TEACH from the [ST2M release](https://github.com/Druthrie/ST2M)
(same [Google Drive](https://drive.google.com/file/d/1_KCzpH6BA-7BnL2_QrkJa2CA_RQUAvWR/view?usp=sharing)
as the dataset) and arrange it as:

```text
data/pretrained_models/BABEL_TEACH/text_mot_match_M10_BABEL_TEACH/model/finest.tar
```

The GloVe word vectors used by the evaluation models are already included in [`glove/`](glove/).

## Pretrained Models

Download the pretrained weights of the released model `grained_text_fuse_film`
(model checkpoint at 50,000 iterations, EMA weights) from
[Google Drive](https://drive.google.com/drive/folders/1cs1ewx46RrevJ70vU7FGLAVcMAiYhqLV?usp=drive_link)
and arrange them as:

```text
checkpoints/BABEL_TEACH/grained_text_fuse_film/
├── opt.txt
├── meta/
│   ├── mean.npy
│   └── std.npy
└── model/
    └── model_50000.tar
```

## Training

All defaults in [`options/train_options.py`](options/train_options.py) already match the released
model, so the following command reproduces the released configuration
(`base_dim=512`, `dim_mults=[2,2,2,2]`, `text_latent_dim=256`, `time_dim=512`,
`batch_size=64`, `num_train_steps=50000`, `lr=2e-4` with exponential decay `0.9` every 5000 steps,
`weight_decay=0.01`, `cond_mask_prob=0.1`, EMA decay `0.9999`, `prediction_type=sample`,
`mul_data_size=2`, `unit_length=4`, `feat_bias=5`, `seed=0`):

```shell
# single GPU (accelerate config for 1 process is provided in 1gpu.yaml)
accelerate launch --config_file 1gpu.yaml tools/train.py \
    --name grained_text_fuse_film \
    --dataset_name BABEL_TEACH \
    --num_train_steps 50000 \
    --model-ema
```

For multi-GPU training (DDP), create an accelerate config with `num_processes` set to your number
of GPUs, e.g.:

```shell
accelerate config   # choose multi-GPU
accelerate launch tools/train.py \
    --name grained_text_fuse_film \
    --dataset_name BABEL_TEACH \
    --num_train_steps 50000 \
    --model-ema
```

Checkpoints are saved to `checkpoints/<dataset_name>/<name>/model/` every `--save_interval`
(default 5000) iterations; training logs go to `log/` and can be viewed with tensorboard.

## Evaluation

Evaluation runs 20 repetitions of generation over the whole test set and reports
Matching Score, R-precision (top-1/2/3), FID, Diversity, Multi-Modality and Transition Score.

```shell
python tools/evaluation.py \
    --opt_path checkpoints/BABEL_TEACH/grained_text_fuse_film/opt.txt \
    --gpu_id 0 \
    --log_name grained_text_fuse_film
```

Results are written to `eval_log/BABEL_TEACH/grained_text_fuse_film.log`.
Use `--which_epoch` to evaluate another checkpoint, and `--gpu_id -1` to run on CPU.

## Inference / Visualization

Generate a motion from per-segment captions, target segment lengths and fine-grained text.
The fine-grained text file must contain **6 short phrases per segment**
(head / torso / left arm / right arm / left leg / right leg), see
[`examples/grained_text_example.txt`](examples/grained_text_example.txt).
The command below reproduces the teaser animation at the top of this page:

```shell
python tools/visualization.py \
    --opt_path checkpoints/BABEL_TEACH/grained_text_fuse_film/opt.txt \
    --text "sit down" "raise the right hand" "crawl" \
    --motion_length 60 40 60 \
    --grained_text_file examples/grained_text_example.txt \
    --result_path result/teaser.gif \
    --npy_path result/teaser.npy \
    --gpu_id 0
```

Notes:

- `--motion_length` gives the expected length (frames, 20 fps) of each segment; the maximum
  supported length is 196 frames.
- `--npy_path` (optional) stores the joint coordinates of each frame, shape `(T, 22, 3)`.
- `--is_slerp` (optional) interpolates a 4-frame gap between segments with slerp for a smoother
  transition, as in <img src="figures/teaser_2.gif" width="60">.
- `--which_epoch`, `--num_inference_steps` (default 10) and `--diffuser_name` (default
  `dpmsolver`, see `config/diffuser_params.yaml`) can be adjusted as needed.

## Results

Results of the released checkpoint `model_50000` on the BABEL_TEACH test set
(20 repetitions, values are mean; ground-truth statistics for reference):

| Method | MS &darr; | R-precision (top-1 / 2 / 3) &uarr; | FID &darr; | Diversity &uarr; | MM &uarr; | Transition &darr; |
|---|---|---|---|---|---|---|
| Ground truth | 3.4479 | 0.5626 / 0.7486 / 0.8343 | 0.0016 | 10.4490 | - | 0.0000 |
| FCMD (ours) | 3.3413 | 0.5925 / 0.7830 / 0.8662 | 0.4823 | 10.6815 | 2.3736 | 0.1272 |

## Acknowledgement

This code is built on top of
[MotionDiffuse](https://github.com/GuyTevet/MotionDiffuse) and
[text-to-motion](https://github.com/EricGuo5513/text-to-motion).
We also thank the authors of
[ST2M](https://github.com/Druthrie/ST2M)
(*Sequential Texts Driven Cohesive Motions Synthesis with Natural Transitions*, ICCV 2023) for the
BABEL_TEACH/STDM datasets, the pretrained evaluation models and parts of the code, and the authors of
[StableMoFusion](https://github.com/Linketic/StableMoFusion)
(*StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework*)
for the diffusion-based motion generation framework whose code we referenced.
