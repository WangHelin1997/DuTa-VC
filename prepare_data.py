import os
import random
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
import h5py
import csv
import re
from num2words import num2words

def get_mel(index, savepath, commands):
    hf_m = h5py.File(os.path.join(savepath, 'mels', str(index)+ '.npy'), 'w')
    hf_w = h5py.File(os.path.join(savepath, 'wavs', str(index)+ '.npy'), 'w')
    for wavpath, textpath in tqdm(commands):
        wav, _ = load(wavpath, sr=22050)
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        hf_w.create_dataset(wavpath.split('/')[-1].replace('.wav', ''), data=wav)
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        hf_m.create_dataset(wavpath.split('/')[-1].replace('.wav', ''), data=log_mel_spectrogram)
        # print(wavpath)
    hf_m.close()
    hf_w.close()

def generate_mel_LibriTTS(savenum = 200):
    audiopath = '/data/lmorove1/hwang258/LibriTTS/LibriTTS'
    textgridpath = '/data/lmorove1/hwang258/LibriTTSCorpusLabel/textgrid'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts'
    os.makedirs(os.path.join(savepath, 'mels'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'wavs'), exist_ok=True)
    count = 0
    debug = False
    commands = []
    for root, dir, files in os.walk(textgridpath):
        for f in tqdm(files):
            if f.endswith('.TextGrid'):
                wavpath = os.path.join(root,f).replace(textgridpath,audiopath).replace('TextGrid', 'wav')
                if os.path.exists(wavpath):
                    count += 1
                    commands.append((wavpath, os.path.join(root, f)))
        if debug:
            if count > 2000:
                break
    print(len(commands))
    splitnum = len(commands) // 200
    newcommands = []
    for i in range(savenum-1):
        onecommands = []
        for j in range(splitnum):
            onecommands.append((commands[i*splitnum+j][0], commands[i*splitnum+j][1]))
        newcommands.append((i, savepath, onecommands))
    onecommands = []
    i = savenum-1
    for j in range(splitnum):
        onecommands.append((commands[i*splitnum+j][0], commands[i*splitnum+j][1]))
    newcommands.append((i, savepath, onecommands))

    with multiprocessing.Pool(processes=50) as pool:
        pool.starmap(get_mel, newcommands)

def read_h5():
    h5_file = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/mels/0.npy'
    hf = h5py.File(h5_file, 'r')
    print(hf.keys())
    n1 = np.array(hf.get('5290_26685_000004_000000'))
    print(n1.shape)
    hf.close()

def get_textgrid(textpath, savepath):
    shutil.copyfile(textpath, os.path.join(savepath, 'textgrids', textpath.split('/')[-1]))
    print(textpath)

def copyfile_LibriTTS():
    audiopath = '/data/lmorove1/hwang258/LibriTTS/LibriTTS'
    textgridpath = '/data/lmorove1/hwang258/LibriTTSCorpusLabel/textgrid'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts'
    os.makedirs(os.path.join(savepath, 'textgrids'), exist_ok=True)
    count = 0
    debug = True
    commands = []
    for root, dir, files in os.walk(textgridpath):
        for f in tqdm(files):
            if f.endswith('.TextGrid'):
                wavpath = os.path.join(root,f).replace(textgridpath,audiopath).replace('TextGrid', 'wav')
                if os.path.exists(wavpath):
                    count += 1
                    commands.append((os.path.join(root, f), savepath))
        if debug:
            if count > 2000:
                break
    print(len(commands))

    with multiprocessing.Pool(processes=50) as pool:
        pool.starmap(get_textgrid, commands)

def cal_avg_mel():
    phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2',
                    'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0',
                    'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
                    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2',
                    'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
                    'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P',
                    'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2',
                    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'sil', 'sp', 'spn']
    phoneme_dict = dict()
    for j, p in enumerate(phoneme_list):
        phoneme_dict[p] = j

    data_dir = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts'
    mels_mode_dict = dict()
    lens_dict = dict()
    for p in phoneme_list:
        lens_dict[p] = []

    for root, dir, files in os.walk(os.path.join(data_dir, 'mels')):
        for f in tqdm(files):
            if f.endswith('.npy'):
                textgrid = os.path.join(root, f).replace('mels', 'textgrids').replace('npy', 'TextGrid')
                t = tgt.io.read_textgrid(textgrid)
                m = np.load(os.path.join(root, f))
                t = t.get_tier_by_name('phones')
                for i in range(len(t)):
                    phoneme = t[i].text
                    start_frame = int(t[i].start_time * 22050.0) // 256
                    end_frame = int(t[i].end_time * 22050.0) // 256 + 1
                    if phoneme not in mels_mode_dict.keys():
                        mels_mode_dict[phoneme] = []
                    mels_mode_dict[phoneme] += [np.round(np.median(m[:, start_frame:end_frame], 1), 1)]
                    lens_dict[phoneme] += [end_frame - start_frame]

    mels_mode = dict()
    for p in phoneme_list:
        if p in mels_mode_dict.keys():
            mels_mode[p] = mode(np.asarray(mels_mode_dict[p]), 0).mode[0]
    del mels_mode_dict
    with open(os.path.join(data_dir, 'mels_mode.pkl'), 'wb') as f:
        pickle.dump(mels_mode, f)
    del lens_dict
    return mels_mode

def generate_one_avg_mel_LibriTTS(mfapath, f, datapath, phoneme_list, mels_mode, savepath):
    h5_file = os.path.join(datapath, f)
    hf = h5py.File(h5_file, 'r')
    allkeys = hf.keys()
    hf_mm = h5py.File(os.path.join(savepath, 'mels_mode', f), 'w')
    hf_p = h5py.File(os.path.join(savepath, 'phonemes', f), 'w')
    for k in tqdm(allkeys):
        m = np.array(hf.get(k))
        textgrid = os.path.join(mfapath, k+'.TextGrid')
        t = tgt.io.read_textgrid(textgrid)
        m_mode = m.copy()
        p = np.full((m.shape[1]), 72, dtype=int)
        t = t.get_tier_by_name('phones')
        for i in range(len(t)):
            phoneme = t[i].text
            start_frame = int(t[i].start_time * 22050.0) // 256
            end_frame = int(t[i].end_time * 22050.0) // 256 + 1
            if end_frame > m_mode.shape[1]:
                end_frame = m_mode.shape[1]
            p[start_frame:end_frame] = np.array([phoneme_list.index(phoneme)] * (end_frame - start_frame))
            m_mode[:, start_frame:end_frame] = np.repeat(np.expand_dims(mels_mode[phoneme], 1),
                                                        end_frame - start_frame, 1)
        hf_mm.create_dataset(k, data=m_mode)
        hf_p.create_dataset(k, data=p)
    hf.close()
    hf_mm.close()
    hf_p.close()

def generate_avg_mel_LibriTTS():
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/mels/'
    mfapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/textgrids/'
    pkl_path = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/mels_mode.pkl'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/'
    with open(pkl_path, 'rb') as f:
        mels_mode = pickle.load(f)
    phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                    'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'sil',
                    'sp', 'spn']
    os.makedirs(os.path.join(savepath, 'mels_mode'), exist_ok=True)
    os.makedirs(os.path.join(savepath, 'phonemes'), exist_ok=True)

    cmds = []
    count = 0
    savenum = 200
    for root, dir, files in os.walk(datapath):
        for f in tqdm(files):
            if f.endswith('.npy'):
                cmds.append((mfapath, f, datapath, phoneme_list, mels_mode, savepath))
    print(len(cmds))
    with multiprocessing.Pool(processes=50) as pool:
        pool.starmap(generate_one_avg_mel_LibriTTS, cmds)

def get_embed(f, datapath, spk_encoder, savepath):
    if not os.path.exists(os.path.join(savepath, f)):
        hf = h5py.File(os.path.join(datapath, f), 'r')
        hf_e = h5py.File(os.path.join(savepath, f), 'w')
        allkeys = hf.keys()
        for k in tqdm(allkeys):
            wav = np.array(hf.get(k))
            wav_preprocessed = spk_encoder.preprocess_wav(wav)
            embed = spk_encoder.embed_utterance(wav_preprocessed)
            hf_e.create_dataset(k, data=embed)
        hf.close()
        hf_e.close()

def generate_emb_LibriTTS():
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/wavs'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/embeds'
    os.makedirs(savepath, exist_ok=True)
    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
    spk_encoder.load_model(enc_model_fpath, device="cpu")
    cmds = []
    for root, dir, files in os.walk(datapath):
        for f in files:
            if f.endswith('.npy'):
                cmds.append((f, datapath, spk_encoder, savepath))
    print(len(cmds))
    random.shuffle(cmds)
    for c in cmds:
        get_embed(c[0], c[1], c[2], c[3])
    
def checkfile(i, filepath):
    t = tgt.io.read_textgrid(filepath)
    t = t.get_tier_by_name('phones')
    spn_found = False
    for i in range(len(t)):
        if t[i].text == 'spn':
            spn_found = True
            break
    if spn_found:
        print(filepath)
        os.remove(filepath)


def checkfile_LibriTTS():
    cmds = []
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/'
    for root, dir, files in os.walk(os.path.join(datapath, 'textgrids')):
        for f in files:
            if f.endswith('.TextGrid'):
                cmds.append((0, os.path.join(root, f)))
    print(len(cmds))
    with multiprocessing.Pool(processes=50) as pool:
        pool.starmap(checkfile, cmds)


def generate_list_LibriTTS():
    test_speakers = ['1401', '2238', '3723', '4014', '5126',
                     '5322', '587', '6415', '8057', '8534']
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/'
    os.makedirs(os.path.join(datapath, 'splits'), exist_ok=True)
    train_list = []
    test_list = []
    for root, dir, files in os.walk(os.path.join(datapath, 'textgrids')):
        for f in files:
            if f.endswith('.TextGrid'):
                if f.split('_')[0] in test_speakers:
                    test_list.append(f.split('.TextGrid')[0])
                else:
                    train_list.append(f.split('.TextGrid')[0])
    savenum = 200
    train_list_save = []
    test_list_save = []
    for i in range(savenum):
        h5_file = os.path.join(datapath, 'mels', str(i)+'.npy')
        hf = h5py.File(h5_file, 'r')
        for k in tqdm(list(hf.keys())):
            if k in train_list:
                train_list_save.append(k+'\t'+str(i))
            elif k in test_list:
                test_list_save.append(k+'\t'+str(i))
        hf.close()
    with open(os.path.join(datapath, 'splits', 'train.txt'), 'w') as f:
        for line in train_list_save:
            f.write(line)
            f.write('\n')
    with open(os.path.join(datapath, 'splits', 'test.txt'), 'w') as f:
        for line in test_list_save:
            f.write(line)
            f.write('\n')

def generate_alllist_LibriTTS():
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/'
    train_list = []
    for root, dir, files in os.walk(os.path.join(datapath, 'textgrids')):
        for f in files:
            if f.endswith('.TextGrid'):
                train_list.append(f.split('.TextGrid')[0])
    savenum = 200
    train_list_save = []
    for i in range(savenum):
        h5_file = os.path.join(datapath, 'mels', str(i)+'.npy')
        hf = h5py.File(h5_file, 'r')
        for k in tqdm(list(hf.keys())):
            if k in train_list:
                train_list_save.append(k+'\t'+str(i))
        hf.close()
    with open(os.path.join(datapath, 'all.txt'), 'w') as f:
        for line in train_list_save:
            f.write(line)
            f.write('\n')

def cal_avg_phonemetime_LibriTTS():
    mfapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/textgrids/'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/libristts/'
    phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2',
                    'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0',
                    'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
                    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2',
                    'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
                    'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P',
                    'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2',
                    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'sil', 'sp', 'spn']
    phoneme_dict = {}
    for p in phoneme_list:
        phoneme_dict[p] = {'duration':0., 'number':0}
    for root, dir, files in os.walk(mfapath):
        for f in tqdm(files):
            if f.endswith('.TextGrid'):
                t = tgt.io.read_textgrid(os.path.join(root, f))
                t = t.get_tier_by_name('phones')
                for i in range(len(t)):
                    phoneme = t[i].text
                    phoneme_dict[phoneme]['duration'] += t[i].end_time - t[i].start_time
                    phoneme_dict[phoneme]['number'] += 1
    for p in phoneme_list:
        if phoneme_dict[p]['number'] > 0:
            phoneme_dict[p]['avg_duration'] = float(phoneme_dict[p]['duration'] / phoneme_dict[p]['number'])
        else:
            phoneme_dict[p]['avg_duration'] = 0.
    print(phoneme_dict)
    with open(os.path.join(savepath, 'phonemes.pkl'), 'wb') as f:
        pickle.dump(phoneme_dict, f)

def prepare_lab_aty():
    spks = ['0024', '0025', '0026']
    wavpath = '/scratch4/lmorove1/hwang258/data/atypicalspeech/atypicalspeech/Audios'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/data/aty/'
    metapath = '/scratch4/lmorove1/hwang258/data/atypicalspeech/atypicalspeech/excel_out_slurp_metadata_named.csv'
    lines = []
    with open(metapath, mode ='r')as file:
        csvFile = csv.reader(file) 
        for line in csvFile: 
            lines.append(line)
    lines = lines[1:]
    metadict = {}
    for line in lines:
        metadict[line[2]] = line[3]
    for spk in spks:
        os.makedirs(os.path.join(savepath, spk, 'wavs'), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(wavpath, spk)):
            for f in tqdm(files):
                if f.endswith('.wav'):
                    name = 'XXXX_' + f.split('_', 1)[-1]
                    if name in metadict.keys():
                        shutil.copyfile(os.path.join(root, f), os.path.join(savepath, spk, 'wavs', f))
                        text = metadict[name]
                        text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
                        text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)
                        text = text.replace("@", " at ")
                        text = text.replace("#", " hashtag ")
                        text = text.replace(",", "")
                        text = text.replace(".", "")
                        text = re.sub(" +", " ", text)
                        text_ = [num2words(int(s)) if s.isdigit() else s for s in text.split()]
                        text = ''
                        for s in text_:
                            text += s + ' '
                        text = text[:-1]
                        text = re.sub(r'[^\w\s]', '', text.upper())
                        with open(os.path.join(savepath, spk, 'wavs', f.replace('.wav','.lab')), 'w') as fi:
                            fi.write(text)

# exclude utterances where MFA couldn't recognize some words
def exclude_spn(textgrid):
    t = tgt.io.read_textgrid(textgrid)
    t = t.get_tier_by_name('phones')
    spn_found = False
    for i in range(len(t)):
        if t[i].text == 'spn':
            spn_found = True
            break
    if not spn_found:
        return True
    return False

def makedata_aty():
    txtpath = '/scratch4/lmorove1/hwang258/data/atypicalspeech/atypicalspeech/textgrids'
    audiopath = '/scratch4/lmorove1/hwang258/data/atypicalspeech/atypicalspeech/Audios_22050'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data'
    allspks = ['0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014',
        '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
        '0024', '0025', '0026']
    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'wavs', spk), exist_ok=True)
        os.makedirs(os.path.join(savepath, 'textgrids', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(txtpath, spk)):
            for f in tqdm(files):
                if f.endswith('.TextGrid'):
                    if exclude_spn(os.path.join(root, f)):
                        shutil.copyfile(os.path.join(txtpath, spk, f), os.path.join(savepath, 'textgrids', spk, f))
                        shutil.copyfile(os.path.join(audiopath, spk, f.replace('.TextGrid', '.wav')), os.path.join(savepath, 'wavs', spk, f.replace('.TextGrid', '.wav')))

def get_mel_atypical(wav_path, save_path):
    try:
        wav, _ = load(wav_path, sr=22050)
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        with open(save_path, 'wb') as f:
            np.save(f, log_mel_spectrogram)
    except:
        print(wav_path)


def load_data_aty():
    audiopath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/wavs'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/'
    allspks = ['0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014',
        '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
        '0024', '0025', '0026']

    commands = []
    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'mels', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(audiopath, spk)):
            for f in files:
                if f.endswith('.wav'):
                    commands.append((os.path.join(root, f), os.path.join(savepath, 'mels', spk, f.replace('.wav', '.npy'))))
    print(len(commands))
    with multiprocessing.Pool(processes=40) as pool:
        pool.starmap(get_mel_atypical, commands)


def generate_avg_aty():
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/'
    mfapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/textgrids'
    pkl_path = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/libritts_data/mels_mode.pkl'
    allspks = ['0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014',
        '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
        '0024', '0025', '0026']
    with open(pkl_path, 'rb') as f:
        mels_mode = pickle.load(f)

    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'mels_mode', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(savepath, 'mels', spk)):
            for f in files:
                if f.endswith('.npy'):
                    textgrid = os.path.join(mfapath, spk, f.replace('.npy', '.TextGrid'))
                    t = tgt.io.read_textgrid(textgrid)
                    m = np.load(os.path.join(root, f))
                    m_mode = np.full((m.shape[0], m.shape[1]), np.log(1e-5))
                    t = t.get_tier_by_name('phones')
                    for i in range(len(t)):
                        phoneme = t[i].text
                        start_frame = int(t[i].start_time * 22050.0) // 256
                        end_frame = int(t[i].end_time * 22050.0) // 256 + 1
                        if end_frame > m_mode.shape[1]:
                            end_frame = m_mode.shape[1]
                        print(f, phoneme, m_mode.shape, mels_mode[phoneme].shape, start_frame, end_frame)
                        m_mode[:, start_frame:end_frame] = np.repeat(np.expand_dims(mels_mode[phoneme], 1),
                                                                    end_frame - start_frame, 1)
                    np.save(os.path.join(savepath, 'mels_mode', spk, f.replace('.npy', '_avgmel.npy')), m_mode)


def get_embed(wav_path, spk_encoder, savepath):
    if not os.path.exists(savepath):
        wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
        embed = spk_encoder.embed_utterance(wav_preprocessed)
        np.save(savepath, embed)
        # print(savepath)

def generate_emb_aty():
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/wavs/'
    savepath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/'
    allspks = ['0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014',
        '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
        '0024', '0025', '0026']
    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
    spk_encoder.load_model(enc_model_fpath, device="cpu")
    cmds = []
    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'embeds', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(datapath, spk)):
            for f in files:
                if f.endswith('.wav'):
                    savename = os.path.join(savepath, 'embeds', spk, f.replace('.wav', '.npy'))
                    cmds.append((os.path.join(root, f), spk_encoder, savename))
    print(len(cmds))
    random.shuffle(cmds)
    for c in tqdm(cmds):
        get_embed(c[0], c[1], c[2])


def cal_mean_std_aty():
    datapath = '/data/lmorove1/hwang258/Speech-Backbones/DiffVC/aty_data/'
    allspks = ['0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014',
        '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
        '0024', '0025', '0026']
    for spk in allspks:
        os.makedirs(os.path.join(datapath, 'stats', spk), exist_ok=True)
        data_list = []
        for root, dir, files in os.walk(os.path.join(datapath, 'mels', spk)):
            for f in files:
                if f.endswith('.npy'):
                    data_list.append(os.path.join(root, f))
        scp_list_source = data_list
        print(len(scp_list_source))
        feaNormCal = []
        for s in tqdm(scp_list_source):
            x = np.load(s).transpose(1, 0)
            feaNormCal.extend(x)
        nFrame = np.shape(feaNormCal)[0]
        print(nFrame)
        feaMean = np.mean(feaNormCal, axis=0)
        for i in range(nFrame):
            if i == 0:
                feaStd = np.square(feaNormCal[i] - feaMean)
            else:
                feaStd += np.square(feaNormCal[i] - feaMean)
        feaStd = np.sqrt(feaStd / nFrame)
        result = np.vstack((feaMean, feaStd))
        np.savetxt(os.path.join(datapath, 'stats', spk, 'global_mean_var.txt'), result)

if __name__ == "__main__":
    print('test')
    # copyfile_LibriTTS()
    # checkfile_LibriTTS()
    # generate_mel_LibriTTS()
    # read_h5()
    # cal_avg_mel()
    # generate_avg_mel_LibriTTS()
    # generate_emb_LibriTTS()
    # generate_list_LibriTTS()
    # cal_avg_phonemetime_LibriTTS()
    # prepare_lab_aty()
    # generate_alllist_LibriTTS()
    # makedata_aty()
    load_data_aty()
    generate_avg_aty()
    generate_emb_aty()
    cal_mean_std_aty()
