# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import random
import numpy as np
import torch
from params import seed as random_seed
from params import n_mels, train_frames


class ATYDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, spk):
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.speakers = [spk]
        stats = np.loadtxt(os.path.join(data_dir, 'stats', spk, 'global_mean_var.txt'))
        self.mean = stats[0]
        self.std = stats[1]
        self.train_info = []
        self.valid_info = []
        self.read_info()
        for spk in self.speakers:
            mel_ids = []
            for root, dirs, files in os.walk(os.path.join(self.mel_dir, spk)):
                for f in files:
                    if f.endswith('.npy'):
                        mel_ids.append(f.split('.npy')[0])
            self.train_info += [(m, spk) for m in mel_ids]

        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def read_info(self):
        allnames = []
        for dys in self.speakers:
            for root, dirs, files in os.walk(os.path.join(self.mel_dir, dys)):
                for f in files:
                    if f.endswith('.npy'):
                        allnames.append(f.split('.npy')[0])
        random.shuffle(allnames)

    def mean_var_norm(self, x):
        x = (x - self.mean[:, None]) / self.std[:, None]
        return x

    def inv_mean_var_norm(self, x):
        x = (x * self.std[:, None]) + self.mean[:, None]
        return x

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '.npy')
        mels = np.load(mel_path)
        mels = self.mean_var_norm(mels)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return pairs

class ATYDecBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed}