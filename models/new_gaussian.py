from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler, DEISMultistepScheduler
import torch
import yaml
from tqdm import tqdm


class DiffusePipeline(object):

    def __init__(self, opt, encoder, diffuser_name, num_inference_steps, device, torch_dtype=torch.float16, overlap=6):
        self.device = device
        self.torch_dtype = torch_dtype
        self.diffuser_name = diffuser_name
        self.num_inference_steps = num_inference_steps
        self.encoder = encoder
        self.overlap = overlap
        self.opt = opt
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=opt.diffusion_steps,
                                             beta_schedule=opt.beta_schedule,
                                             variance_type="fixed_small",
                                             prediction_type=opt.prediction_type,
                                             clip_sample=False)

        # Load parameters from YAML file
        with open('./config/diffuser_params.yaml', 'r') as yaml_file:
            diffuser_params = yaml.safe_load(yaml_file)

        # Select diffusion parameters based on diffuser_name
        if diffuser_name in diffuser_params:
            params = diffuser_params[diffuser_name]
            scheduler_class_name = params['scheduler_class']
            additional_params = params['additional_params']

            # align training parameters
            additional_params['num_train_timesteps'] = opt.diffusion_steps
            additional_params['beta_schedule'] = opt.beta_schedule
            additional_params['prediction_type'] = opt.prediction_type

            try:
                scheduler_class = globals()[scheduler_class_name]
            except KeyError:
                raise ValueError(f"Class '{scheduler_class_name}' not found.")

            self.scheduler = scheduler_class(**additional_params)
        else:
            raise ValueError(f"Unsupported diffuser_name: {diffuser_name}")

    def stitch(self, all_output, m_lens):
        weight = (torch.arange(self.overlap, device=all_output.device) + 0.5) / self.overlap
        N, B, T, D = all_output.shape
        flag = 196 - self.overlap
        for j in range(B):
            lap = all_output[0, j, :self.overlap]
            for i in range(N):
                lap_motion = lap * (1 - weight[:, None]) + all_output[i, j, :self.overlap] * weight[:, None]
                all_output[i, j, :self.overlap] = lap_motion
                m_len = m_lens[i][j]
                left = 0 if m_len < flag else m_len - flag
                lap = all_output[i, j, m_len - left:m_len + self.overlap - left]
        return all_output

    def generate_ST2M(self, caption, m_lens, grained_text, dim_pose):
        self.encoder.eval()
        if isinstance(caption[0], str):
            B = 1
        else:
            B = len(caption[0])
        N = len(caption)
        T = self.encoder.num_frames
        sample = torch.randn((N, B, T, dim_pose), device=self.device)
        all_output = torch.zeros((N, B, T, dim_pose), device=self.device)
        self.scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = [torch.tensor([t] * B, device=self.device).long() for t in self.scheduler.timesteps]
        for t in tqdm(timesteps):
            for i in range(N):
                shape = (B, T, dim_pose)
                sample_i = sample[i]
                if i == 0:
                    his_movement = torch.zeros(shape, device=self.device)
                    his_caption = ['start'] * B
                    his_text_feature = self.encoder.encode_text(his_caption, self.device)
                xf_out = self.encoder.encode_text(caption[i], self.device)
                grained_text_feature = self.encoder.encode_grain_text(grained_text[i], self.device)
                with torch.no_grad():
                    if getattr(self.encoder, 'cond_mask_prob', 0) > 0:
                        output = self.encoder.forward_with_cfg(sample_i, t, i, m_lens, caption, his_movement,
                                                               his_text_feature, xf_out, grained_text_feature)
                    else:
                        output = self.encoder(sample_i, t, i, m_lens, caption, his_movement,
                                              his_text_feature, xf_out, grained_text_feature)
                his_text_feature = xf_out
                his_movement = output
                all_output[i] = output
            stitch_output = self.stitch(all_output, m_lens)
            sample = self.scheduler.step(stitch_output, t[0], sample).prev_sample
        return all_output
