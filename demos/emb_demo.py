import os
import random
import pytsmod as tsm
import tgt
from scipy.stats import mode
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)
import pickle
import numpy as np
import torch
use_gpu = torch.cuda.is_available()
import sys
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
import multiprocessing
import shutil
from tqdm import tqdm
import soundfile as sf
import json
from numpy.linalg import norm
import pandas as pd

def get_embed(wav_path, spk_encoder, savepath):
    if not os.path.exists(savepath):
        wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
        embed = spk_encoder.embed_utterance(wav_preprocessed)
        np.save(savepath, embed)
        # print(savepath)

def generate_emb_GE(source_dys, target_dys):
    datapath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/results_allaugdata'
    savepath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/embs_demo_all/'
    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
    spk_encoder.load_model(enc_model_fpath, device="cpu")
    cmds = []
    for root, dir, files in os.walk(os.path.join(datapath, target_dys)):
        for f in files:
            if f.endswith('.wav'):
                if f.startswith(source_dys):
                    savename = os.path.join(savepath, source_dys+'to'+target_dys, f).replace('.wav', '.npy')
                    os.makedirs(os.path.join(savepath, source_dys+'to'+target_dys), exist_ok=True)
                    cmds.append((os.path.join(root, f), spk_encoder, savename))
    random.shuffle(cmds)
    cmds = cmds[:20]
    for c in tqdm(cmds):
        get_embed(c[0], c[1], c[2])

def cal_similarity(source_dys, target_dys):
    generate_emb_GE(source_dys, target_dys)
    datapath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/avgmel_data/embeds'
    embpath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/embs_demo_all/'
    source_embs = []
    target_embs = []
    generated_embs = []
    for root, dirs, files in os.walk(os.path.join(datapath, source_dys)):
        for f in files:
            if f.endswith('.npy'):
                source_embs.append(np.load(os.path.join(root, f)))
    for root, dirs, files in os.walk(os.path.join(datapath, target_dys)):
        for f in files:
            if f.endswith('.npy'):
                target_embs.append(np.load(os.path.join(root, f)))
    for root, dirs, files in os.walk(os.path.join(embpath, source_dys+'to'+target_dys)):
        for f in files:
            if f.endswith('.npy'):
                generated_embs.append(np.load(os.path.join(root, f)))
    source_embs = np.array(source_embs)
    target_embs = np.array(target_embs)
    generated_embs = np.array(generated_embs)
    source_embs = np.mean(source_embs, 0)
    target_embs = np.mean(target_embs, 0)
    generated_embs = np.mean(generated_embs, 0)
    cos_sg = np.dot(source_embs, generated_embs) / (norm(source_embs) * norm(generated_embs))
    cos_tg = np.dot(target_embs, generated_embs) / (norm(target_embs) * norm(generated_embs))
    cos_st = np.dot(target_embs, source_embs) / (norm(target_embs) * norm(source_embs))
    print(source_dys, target_dys, cos_st, cos_sg, cos_tg)
    return source_dys, target_dys, cos_st, cos_sg, cos_tg

dysspks = ['F02', 'F03', 'F04', 'F05', 'M01', 'M04', 'M05', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M14', 'M16']
ctrlspks = ['CF02', 'CF03', 'CF04', 'CF05', 'CM01', 'CM04', 'CM05', 'CM06', 'CM08', 'CM09', 'CM10', 'CM12', 'CM13']

dicts = {'Source':[],
'Target':[],
         'ST':[],
         'SG':[],
         'TG':[]
}
df = pd.DataFrame(dicts)
allst, allsg,alltg=0.,0.,0.
for c in ctrlspks:
    for d in dysspks:
        source_dys, target_dys, cos_st, cos_sg, cos_tg = cal_similarity(c, d)
        df2 = {'Source': source_dys, 'Target': target_dys, 'ST': cos_st, 'SG': cos_sg, 'TG':cos_tg}
        df = df.append(df2, ignore_index=True)
        allsg+=cos_sg
        allst+=cos_st
        alltg+=cos_tg
allsg/=len(dysspks)*len(ctrlspks)
allst/=len(dysspks)*len(ctrlspks)
alltg/=len(dysspks)*len(ctrlspks)
df2 = {'Source': 'All', 'Target': 'All', 'ST': allst, 'SG': allsg, 'TG': alltg}
df = df.append(df2, ignore_index=True)
df = df.round(3)
df.to_excel('SpeakerSimilarity.xlsx')