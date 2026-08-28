from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class st2m_Text2Motion_withpast_DatasetV5(data.Dataset):
    """Dataset for Text2Motion generation task.
    """
    def __init__(self, opt, mean, std, split_file, accelerator=None,w_vectorizer=None, eval_mode=False):
        self.opt = opt
        self.max_length = 20
        self.w_vectorizer = w_vectorizer
        self.eval_mode = eval_mode


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        for name in tqdm(id_list,disable=not accelerator.is_local_main_process if accelerator is not None else False):
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
                        motion_i = motion_i[:length_i-gap]

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
                        num+=1
                        line = line.strip("\n")
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

    def real_len(self):
        return len(self.data_dict)

    def __len__(self):
        return self.real_len()

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion_list, m_length_list, text_list,grained_text = data['motion'], data['length'], data['text'],data['grained_text']

        max_motion_length = self.opt.max_motion_length

        caption = []
        motion = []
        m_length = []
        for i in range(self.opt.mul_data_size):
            motion_i = motion_list[i]
            m_length_i = m_length_list[i]
            text_data_i = text_list[i]
            caption_i, tokens_i = text_data_i['caption'], text_data_i['tokens']
            if m_length_i >= self.opt.max_motion_length:
                idx = random.randint(0, len(motion_i) - max_motion_length)
                motion_i = motion_i[idx: idx + max_motion_length]
            else:
                padding_len = max_motion_length - m_length_i
                D = motion_i.shape[1]
                padding_zeros = np.zeros((padding_len, D))
                motion_i = np.concatenate((motion_i, padding_zeros), axis=0)

            assert len(motion_i) == max_motion_length
            "Z Normalization"
            motion_i = (motion_i - self.mean) / self.std

            caption.append(caption_i)
            motion.append(motion_i[None, :])
            m_length.append(m_length_i)

        motion = np.concatenate(motion, axis=0)
        m_length = np.array(m_length)

        return caption, motion, m_length,grained_text
