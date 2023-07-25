# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import params
from data import ATYDecDataset, ATYDecBatchCollate
from model.vc import DiffVC

n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim

dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size

data_dir = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data'
log_dir = 'logs_dec_aty'
vc_path = 'logs_dec_LT/vc.pt'
allspks = [
    '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', 
    '0014', '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
    '0024', '0025', '0026'
    ]

epochs = 40
batch_size = 32
learning_rate = 5e-5
save_every = 1


def main(dys):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    log_dir_dys = os.path.join(log_dir, dys)
    os.makedirs(log_dir_dys, exist_ok=True)

    print('Initializing data loaders...')
    train_set = ATYDecDataset(data_dir, dys)
    collate_fn = ATYDecBatchCollate()
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              collate_fn=collate_fn, num_workers=16, drop_last=True)
    print(len(train_set))
    print('Initializing and loading models...')
    model = DiffVC(n_mels, channels, filters, heads, layers, kernel,
                   dropout, window_size, enc_dim, spk_dim, use_ref_t,
                   dec_dim, beta_min, beta_max)
    model.load_state_dict(torch.load(vc_path, map_location='cpu'))
    model = model.cuda()
    print('Encoder:')
    print(model.encoder)
    print('Number of parameters = %.2fm\n' % (model.encoder.nparams / 1e6))
    print('Decoder:')
    print(model.decoder)
    print('Number of parameters = %.2fm\n' % (model.decoder.nparams / 1e6))

    print('Initializing optimizers...')
    optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

    print('Start training.')
    torch.backends.cudnn.benchmark = True
    iteration = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch} [iteration: {iteration}]')
        model.train()
        losses = []
        for batch in tqdm(train_loader, total=len(train_set) // batch_size):
            mel, mel_ref = batch['mel1'].cuda(), batch['mel2'].cuda()
            c, mel_lengths = batch['c'].cuda(), batch['mel_lengths'].cuda()
            model.zero_grad()
            loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
            optimizer.step()
            losses.append(loss.item())
            iteration += 1

        losses = np.asarray(losses)
        msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
        print(msg)
        with open(f'{log_dir_dys}/train_dec.log', 'a') as f:
            f.write(msg)

        if epoch % save_every > 0:
            continue

        print('Saving model...\n')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir_dys}/vc.pt")

if __name__ == "__main__":
    for spk in allspks:
        main(spk)