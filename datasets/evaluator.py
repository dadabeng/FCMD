import torch
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.get_opt import get_opt
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
from .evaluator_models import *
import os
import codecs as cs


'''For use of evaluations: generates motions for the test set with the trained model'''


class st2mV13GeneratedDatasetV2_reallen(Dataset):

    def __init__(self, opt, trainer, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)

        print('Loading model: Epoch %03d' % (49))

        all_caption_0 = []
        all_caption_1 = []
        all_m_lens_0 = []
        all_m_lens_1 = []
        all_grained_text_0 = [[], [], [], [], [], []]
        all_grained_text_1 = [[], [], [], [], [], []]
        all_data = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb_0, word_emb_1, pos_ohot_0, pos_ohot_1, caption_0, caption_1, \
                    cap_lens_0, cap_lens_1, motions_0, motions_1, m_lens_0, m_lens_1, tokens_0, tokens_1, \
                    word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens, id_name, grained_text = data
                grained_text[0] = [element[0] for element in grained_text[0]]
                grained_text[1] = [element[0] for element in grained_text[1]]
                all_data.append(data)
                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1

                for t in range(repeat_times):
                    all_m_lens_0.append(m_lens_0)
                    all_m_lens_1.append(m_lens_1-4)
                    all_caption_0.extend(caption_0)
                    all_caption_1.extend(caption_1)
                    for i in range(6):
                        all_grained_text_0[i].append(grained_text[0][i])
                        all_grained_text_1[i].append(grained_text[1][i])
                if is_mm:
                    mm_generated_motions.append(0)

        all_m_lens = torch.tensor([all_m_lens_0, all_m_lens_1], device=opt.device)
        all_caption = [all_caption_0, all_caption_1]

        # Generate all sequences
        with torch.no_grad():
            N = 16
            x = len(all_caption[0])//N
            all_pred_motions = torch.zeros((2,len(all_caption[0]),opt.max_motion_length,opt.dim_pose),device=opt.device)
            for i in range(N):
                before = i*x
                after = (i+1)*x
                grain_text_part = [[],[]]
                if i==N-1:
                    for j in range(6):
                        grain_text_part[0].append(all_grained_text_0[j][before:])
                        grain_text_part[1].append(all_grained_text_1[j][before:])
                    all_pred_motions[:,before:] = trainer.generate_ST2M([sublist[before:] for sublist in all_caption], \
                                                                         all_m_lens[:,before:],grain_text_part, opt.dim_pose)
                else:
                    for j in range(6):
                        grain_text_part[0].append(all_grained_text_0[j][before:after])
                        grain_text_part[1].append(all_grained_text_1[j][before:after])
                    all_pred_motions[:,before:after] = trainer.generate_ST2M([sublist[before:after] for sublist in all_caption], \
                                                                         all_m_lens[:,before:after], grain_text_part, opt.dim_pose)

        cur_idx = 0
        mm_generated_motions = []
        with torch.no_grad():
            for i, datamm in tqdm(enumerate(dataloader)):
                data = all_data[i]
                word_emb_0, word_emb_1, pos_ohot_0, pos_ohot_1, caption_0, caption_1, \
                    cap_lens_0, cap_lens_1, motions_0, motions_1, m_lens_0, m_lens_1, tokens_0, tokens_1, \
                    word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens, id_name, grained_text = data

                tokens_0 = tokens_0[0].split('_')
                tokens_1 = tokens_1[0].split('_')
                tokens = tokens[0].split('_')

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    pred_motions_0 = all_pred_motions[0][cur_idx][0:m_lens_0.item()]
                    pred_motions_1 = all_pred_motions[1][cur_idx][0:m_lens_1.item()]
                    cur_idx += 1
                    assert pred_motions_0.shape[0] == m_lens_0

                    dict_motion = np.concatenate([pred_motions_0.cpu().numpy(), pred_motions_1.cpu().numpy()], axis=0)
                    length_0 = pred_motions_0.shape[0]
                    length_1 = pred_motions_1.shape[0]
                    length = dict_motion.shape[0]

                    if t == 0:
                        sub_dict = {'motion_0': pred_motions_0.cpu().numpy(),
                                    'motion_1': pred_motions_1.cpu().numpy(),
                                    'length_0': length_0,
                                    'length_1': length_1,
                                    'cap_len_0': cap_lens_0[0].item(),
                                    'cap_len_1': cap_lens_1[0].item(),
                                    'caption_0': caption_0[0],
                                    'caption_1': caption_1[0],
                                    'tokens_0': tokens_0,
                                    'tokens_1': tokens_1,
                                    'motion': dict_motion,
                                    'length': length,
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens,
                                    'id_name': id_name
                                    }
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion_0': pred_motions_0.cpu().numpy(),
                            'motion_1': pred_motions_1.cpu().numpy(),
                            'length_0': length_0,
                            'length_1': length_1,
                            'motion': dict_motion,
                            'length': length
                        })

                if is_mm:
                    mm_generated_motions.append({'caption_0': caption_0[0],
                                                 'caption_1': caption_1[0],
                                                 'tokens_0': tokens_0,
                                                 'tokens_1': tokens_1,
                                                 'cap_len_0': cap_lens_0[0].item(),
                                                 'cap_len_1': cap_lens_1[0].item(),
                                                 'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion_0, m_length_0, caption_0, tokens_0 = data['motion_0'], data['length_0'], data['caption_0'], data[
            'tokens_0']
        motion_1, m_length_1, caption_1, tokens_1 = data['motion_1'], data['length_1'], data['caption_1'], data[
            'tokens_1']
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len_0 = data['cap_len_0']
        sent_len_1 = data['cap_len_1']
        sent_len = data['cap_len']
        id_name = data['id_name']

        pos_one_hots_0 = []
        word_embeddings_0 = []
        for token in tokens_0:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots_0.append(pos_oh[None, :])
            word_embeddings_0.append(word_emb[None, :])
        pos_one_hots_0 = np.concatenate(pos_one_hots_0, axis=0)
        word_embeddings_0 = np.concatenate(word_embeddings_0, axis=0)

        pos_one_hots_1 = []
        word_embeddings_1 = []
        for token in tokens_1:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots_1.append(pos_oh[None, :])
            word_embeddings_1.append(word_emb[None, :])
        pos_one_hots_1 = np.concatenate(pos_one_hots_1, axis=0)
        word_embeddings_1 = np.concatenate(word_embeddings_1, axis=0)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length_0 < self.opt.max_motion_length:
            motion_0 = np.concatenate([motion_0,
                                       np.zeros((self.opt.max_motion_length - m_length_0, motion_0.shape[1]))
                                       ], axis=0)

        if m_length_1 < self.opt.max_motion_length:
            motion_1 = np.concatenate([motion_1,
                                       np.zeros((self.opt.max_motion_length - m_length_1, motion_1.shape[1]))
                                       ], axis=0)

        if m_length < self.opt.max_motion_length * 2:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length * 2 - m_length, motion.shape[1]))
                                     ], axis=0)
        grained_text = 1

        return word_embeddings_0, word_embeddings_1, pos_one_hots_0, pos_one_hots_1, caption_0, caption_1, \
            sent_len_0, sent_len_1, motion_0, motion_1, m_length_0, m_length_1, '_'.join(tokens_0), '_'.join(tokens_1), \
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name, grained_text


'''For use of evaluations'''


class st2m_Text2Motion_withpast_Dataset_evalV2(Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.max_length = 20
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = opt.max_motion_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        for name in tqdm(id_list):
            try:
                motion = []
                motion_length = []

                for i in range(self.opt.mul_data_size):
                    motion_i = np.load(pjoin(opt.motion_dir, name + '_C%03d' % i + '.npy'))  # (frame,263)
                    length_i = len(motion_i)
                    gap = length_i % self.opt.unit_length

                    if i == 0:
                        motion_i = motion_i[gap:]
                    else:
                        motion_i = motion_i[:length_i - gap]

                    motion.append(motion_i)
                    motion_length.append(len(motion_i))

                text_data = []

                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_data.append(text_dict)
                grained_text_data = []
                grained_text = []
                with cs.open(pjoin(opt.grained_text_dir, name + '.txt')) as f:
                    num = 0
                    for line in f.readlines():
                        num += 1
                        line.strip("\n")
                        grained_text.append(line)
                        if num == 6:
                            grained_text_data.append(grained_text)
                            grained_text = []
                            num = 0
                data_dict[name] = {'motion': motion,
                                   'length': motion_length,
                                   'text': text_data,
                                   'grained_text': grained_text_data}
                new_name_list.append(name)
                length_list.append(max(motion_length[0], motion_length[1]))

            except:
                # Some motion may not exist in KIT dataset
                print(name)
                pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
        np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = item
        id_name = self.name_list[idx]
        data = self.data_dict[self.name_list[idx]]
        motion_list, m_length_list, text_list, grained_text = data['motion'], data['length'], data['text'], data[
            'grained_text']

        motion = np.concatenate(motion_list, axis=0)
        m_length = m_length_list[0] + m_length_list[1]
        caption = text_list[0]['caption'] + ' --> ' + text_list[1]['caption']
        tokens = text_list[0]['tokens'] + text_list[1]['tokens']

        if len(tokens) < self.opt.max_text_len * 2:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len * 2 + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len * 2]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        motion = (motion - self.mean) / self.std
        motion = np.concatenate((motion,
                                 np.zeros((self.max_motion_length * 2 - m_length, motion.shape[1]))
                                 ), axis=0)

        for i in range(self.opt.mul_data_size):
            motion_i = motion_list[i]
            m_length_i = m_length_list[i]
            text_data_i = text_list[i]
            caption_i, tokens_i = text_data_i['caption'], text_data_i['tokens']
            caption_i_split = caption_i.split(' ')
            if len(caption_i_split) > self.opt.max_text_len + 2:
                caption_i = caption_i_split[0]
                for j in range(self.opt.max_text_len + 1):
                    caption_i = caption_i + ' ' + caption_i_split[j + 1]

            if len(tokens_i) < self.opt.max_text_len:
                # pad with "unk"
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
                tokens_i = tokens_i + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len_i)
            else:
                # crop
                tokens_i = tokens_i[:self.opt.max_text_len]
                tokens_i = ['sos/OTHER'] + tokens_i + ['eos/OTHER']
                sent_len_i = len(tokens_i)
            pos_one_hots_i = []
            word_embeddings_i = []
            for token in tokens_i:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_i.append(pos_oh[None, :])
                word_embeddings_i.append(word_emb[None, :])
            pos_one_hots_i = np.concatenate(pos_one_hots_i, axis=0)
            word_embeddings_i = np.concatenate(word_embeddings_i, axis=0)

            motion_i = (motion_i - self.mean) / self.std
            motion_i = np.concatenate((motion_i,
                                       np.zeros((self.max_motion_length - m_length_i, motion_i.shape[1]))
                                       ), axis=0)

            if i == 0:
                word_embeddings_0 = word_embeddings_i
                pos_one_hots_0 = pos_one_hots_i
                caption_0 = caption_i
                sent_len_0 = sent_len_i
                motion_0 = motion_i
                m_length_0 = m_length_i
                tokens_0 = tokens_i
            elif i == 1:
                word_embeddings_1 = word_embeddings_i
                pos_one_hots_1 = pos_one_hots_i
                caption_1 = caption_i
                sent_len_1 = sent_len_i
                motion_1 = motion_i
                m_length_1 = m_length_i
                tokens_1 = tokens_i

        return word_embeddings_0, word_embeddings_1, pos_one_hots_0, pos_one_hots_1, caption_0, caption_1, \
            sent_len_0, sent_len_1, motion_0, motion_1, m_length_0, m_length_1, '_'.join(tokens_0), '_'.join(tokens_1), \
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name, grained_text


def ST2M_get_dataset_motion_loader(opt_path, batch_size, device):
    opt = get_opt(opt_path, device)

    # Configurations of STDM dataset and BABEL_TEACH dataset is almost the same
    if opt.dataset_name == 'BABEL_TEACH':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test_2000.txt')

        dataset = st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True)
    elif opt.dataset_name == 'STDM':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test_793.txt')
        dataset = st2m_Text2Motion_withpast_Dataset_evalV2(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


class MMGeneratedDatasetV2(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            if len(motion) < self.opt.max_motion_length * 2:
                motion = np.concatenate([motion,
                                         np.zeros((self.opt.max_motion_length * 2 - len(motion), motion.shape[1]))
                                         ], axis=0)
            motion = motion[None, :]
            motions.append(motion)

        m_lens = np.array(m_lens, dtype=int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]

        return motions, m_lens


def get_motion_loader(opt, batch_size, trainer, ground_truth_dataset, mm_num_samples, mm_num_repeats):
    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 'BABEL_TEACH' or opt.dataset_name == 'STDM':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    dataset = st2mV13GeneratedDatasetV2_reallen(opt, trainer, ground_truth_dataset, w_vectorizer, mm_num_samples,
                                                mm_num_repeats)
    mm_dataset = MMGeneratedDatasetV2(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader


def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(
        pjoin('./data/pretrained_models', opt.dataset_name, 'text_mot_match_M10_' + opt.dataset_name, 'model',
              'finest.tar'),
        map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 'BABEL_TEACH' or opt.dataset_name == 'STDM':
            opt.dim_pose = 263
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            org_idx = np.argsort(align_idx.tolist()).copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = torch.div(m_lens, self.opt.unit_length, rounding_mode='trunc')
            motion_embedding = self.motion_encoder(movements, m_lens)
            motion_embedding = motion_embedding[org_idx]

            align_idx = np.argsort(cap_lens.data.tolist())[::-1].copy()
            org_idx = np.argsort(align_idx.tolist()).copy()
            word_embs = word_embs[align_idx]
            pos_ohot = pos_ohot[align_idx]
            cap_lens = cap_lens[align_idx]

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[org_idx]
        return text_embedding, motion_embedding

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            org_idx = np.argsort(align_idx.tolist()).copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = torch.div(m_lens, self.opt.unit_length, rounding_mode='trunc')
            motion_embedding = self.motion_encoder(movements, m_lens)
            motion_embedding = motion_embedding[org_idx]
        return motion_embedding
