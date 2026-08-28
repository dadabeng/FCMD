import torch
import time
import torch.optim as optim
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin

from diffusers import DDPMScheduler
from torch.utils.tensorboard import SummaryWriter
import time
import sys
import os
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from utils.model_load import load_model_weights

class DDPMTrainer(object):

    def __init__(self, args, model,accelerator, model_ema=None):
        self.opt = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.model = model
        self.diffusion_steps = args.diffusion_steps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps= self.diffusion_steps,
            beta_schedule=args.beta_schedule,
            variance_type="fixed_small",
            prediction_type= args.prediction_type,
            clip_sample=False)
        self.model_ema = model_ema
        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')

        accelerator.print('Diffusion_config:\n',self.noise_scheduler.config)

        if self.accelerator.is_main_process:
            starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
            print("Start experiment:", starttime)
            self.writer = SummaryWriter(log_dir=self.opt.log_dir)
            
        self.accelerator.wait_for_everyone()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate) if args.decay_rate>0 else None

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    def clip_norm(self,network_list):
        for network in network_list:
            self.accelerator.clip_grad_norm_(network.parameters(), self.opt.clip_grad_norm) # 0.5 -> 1

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data):
        caption, motions, m_lens, grained_text = batch_data

        motions = motions.detach().to(self.device).float()

        x_start = motions.transpose(0,1)  # [batch,data_size,motion_length,motion_dim]
        B = x_start.shape[1]
        T = x_start.shape[2]
        D = x_start.shape[3]
        self.src_mask = []
        self.prediction = []
        self.target = []
        cur_lens = []
        t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
        self.timesteps = t
        for i in range(self.opt.mul_data_size):
            cur_len = torch.LongTensor([min(T, m_len[i]) for m_len in m_lens]).to(self.device)
            cur_lens.append(cur_len)
            
        for i in range(self.opt.mul_data_size):
            if i==0:
                his_movement = torch.zeros((B,T,D),device = self.device)
            else:
                his_movement = x_start[i-1]
            real_noise = torch.randn_like(x_start[i])
            x_t = self.noise_scheduler.add_noise(x_start[i], real_noise, t)
            self.src_mask.append(self.generate_src_mask(T, cur_lens[i]).to(x_start.device))
            result = self.model(x_t, t, i, cur_lens, caption, his_movement,grained_text=grained_text)
            self.prediction.append(result)
            if self.opt.prediction_type =='sample':
                self.target.append(x_start[i])
            elif self.opt.prediction_type == 'epsilon':
                self.target.append(real_noise)
        self.prediction = torch.cat((self.prediction[0],self.prediction[1]), dim=1)
        self.target = torch.cat((self.target[0],self.target[1]),dim = 1)
        self.src_mask = torch.cat((self.src_mask[0],self.src_mask[1]), dim=1)
    
    def masked_l2(self, a, b, mask, weights):
        
        loss = self.mse_criterion(a, b).mean(dim=-1) # (bath_size, motion_length)
        
        loss = (loss * mask).sum(-1) / mask.sum(-1) # (batch_size, )

        loss = (loss * weights).mean()
        
        return loss
    
    def backward_G(self):
        loss_logs = OrderedDict({})
        mse_loss_weights = torch.ones_like(self.timesteps)
        loss_logs['loss_mot_rec'] = self.masked_l2(self.prediction, self.target, self.src_mask, mse_loss_weights)       
        self.loss = loss_logs['loss_mot_rec']

        return loss_logs

    def update(self):
        self.zero_grad([self.optimizer])
        loss_logs = self.backward_G()
        self.accelerator.backward(self.loss)
        self.clip_norm([self.model])
        self.step([self.optimizer])

        return loss_logs
    
    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def train_mode(self):
        self.model.train()
        if self.model_ema:
            self.model_ema.train()

    def eval_mode(self):
        self.model.eval()
        if self.model_ema:
            self.model_ema.eval()

    def save(self, file_name,total_it):
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if self.model_ema:
            state["model_ema"] = self.accelerator.unwrap_model(self.model_ema).module.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        if self.model_ema:
            self.model_ema.load_state_dict(checkpoint["model_ema"], strict=True)
        self.model.load_state_dict(checkpoint['encoder'], strict=True)
       
        return checkpoint.get('total_it', 0)
    
    
    def train(self, train_dataset):
        
        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)
            it =load_model_weights(self.model, model_path, use_ema=not self.opt.no_ema)
            self.accelerator.print(f'continue train from  {it} iters in {model_path}')
        start_time = time.time()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size= self.opt.batch_size,
            drop_last=True,
            num_workers=4,
            shuffle=True,
            persistent_workers=True)
        
        logs = OrderedDict()
        self.dataset = train_dataset
        
        self.model,self.mse_criterion,self.optimizer,train_loader, self.model_ema = \
        self.accelerator.prepare(self.model,self.mse_criterion,self.optimizer,train_loader,self.model_ema)

        num_epochs = (self.opt.num_train_steps-it)//len(train_loader)  + 1 
        self.accelerator.print(f'need to train for {num_epochs} epochs....')
        
        for epoch in range(0, num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                it += 1

                if self.model_ema and it % self.opt.model_ema_steps == 0:
                    self.accelerator.unwrap_model(self.model_ema).update_parameters(self.model)

                # update logger
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                
                if it % self.opt.log_every == 0 :                   
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(self.accelerator,start_time, it, mean_loss, epoch, inner_iter=i)
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("loss_mot",mean_loss['loss_mot_rec'],it)
                    self.accelerator.wait_for_everyone()
                
                if it % self.opt.save_interval == 0 and self.accelerator.is_main_process:
                    self.save(pjoin(self.opt.model_dir, 'model_'+str(it)+'.tar'), it)
                self.accelerator.wait_for_everyone()


                if (self.scheduler is not None) and (it % self.opt.update_lr_steps == 0) :
                    self.scheduler.step()

        # Save the last checkpoint if it wasn't already saved.
        if it % self.opt.save_interval != 0 and self.accelerator.is_main_process:
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)

        self.accelerator.wait_for_everyone()
        self.accelerator.print('FINISH')

 