import argparse
import json
import os
import numpy as np
import pytsmod as tsm
import torchaudio
import torch
use_gpu = torch.cuda.is_available()
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC
import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
import pickle
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from tqdm import tqdm

def get_mel(wav_path, ratio=1.0, mode=None):
    # mode: tempo, speed or None
    wav, _ = load(wav_path, sr=22050)
    if mode == 'tempo':
        wav = tsm.wsola(wav, ratio)
    elif mode == 'speed':
        wav = librosa.effects.time_stretch(wav, rate=1./ratio)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i+2*w+1])
        y[i] = min(x[i+w+1], med)
    return y

def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


def count_dups(nums):
    element = []
    freque = []
    if not nums:
        return element
    running_count = 1
    for i in range(len(nums)-1):
        if nums[i] == nums[i+1]:
            running_count += 1
        else:
            freque.append(running_count)
            element.append(nums[i])
            running_count = 1
    freque.append(running_count)
    element.append(nums[i+1])
    return element, freque

def inference(args, dys, generator, hifigan_universal, src_path, tgt_path, save_path, mean, std, emb):
    with open(os.path.join(args.phoneme_uaspeech, dys+'_phonemes.pkl'), 'rb') as f:
        ua_dict = pickle.load(f)
    phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                    'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'sil',
                    'sp', 'spn']
    os.makedirs(save_path, exist_ok=True)

    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    mel_target = mel_target.cuda()
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
    mel_target_lengths = mel_target_lengths.cuda()
    embed_target = torch.from_numpy(emb).float().unsqueeze(0)
    embed_target = embed_target.cuda()

    phoneme_logits = np.load(os.path.join(args.phoneme_dir, src_path.split('/')[-1].split('.')[0]+'_phonemes.npy'))
    allp, freque = count_dups(list(phoneme_logits))
    num = 0
    duration = 0.
    duration_gt = 0.
    for i, px in enumerate(allp):
        if px != 69:
            phoneme = phoneme_list[px]
            duration += freque[i]*256
            duration_gt += ua_dict[phoneme]['avg_duration']* 22050.0
            num += 1
    ratio = max(duration_gt / duration, args.fast_ratio)
    # print(duration/num, duration_gt/num, ratio)

    mel_source_tempo = get_mel(src_path, ratio, 'tempo')
    mel_source_tempo = (mel_source_tempo - mean[:, None]) / std[:, None]
    mel_source_tempo = torch.from_numpy(mel_source_tempo).float().unsqueeze(0)
    mel_source_tempo = mel_source_tempo.cuda()
    mel_source_lengths_tempo = torch.LongTensor([mel_source_tempo.shape[-1]])
    mel_source_lengths_tempo = mel_source_lengths_tempo.cuda()

    _, mel_modified = generator(mel_source_tempo, mel_source_lengths_tempo, mel_target, mel_target_lengths, embed_target,
                                          n_timesteps=100, mode='ml')

    mel_synth_np_modified = mel_modified.cpu().detach().squeeze().numpy()
    mel_synth_np_modified = (mel_synth_np_modified * std[:, None]) + mean[:, None]
    mel_modified = torch.from_numpy(mel_spectral_subtraction(mel_synth_np_modified, mel_synth_np_modified, smoothing_window=1)).float().unsqueeze(
        0)
    mel_modified = mel_modified.cuda()
    # converted speech modified
    with torch.no_grad():
        audio = hifigan_universal.forward(mel_modified).cpu().squeeze().reshape(1, -1)
    torchaudio.save(os.path.join(save_path, src_path.split('/')[-1]), audio, 22050)

def get_avg_emb(emb_dir):
    allembs = []
    for root, dirs, files in os.walk(emb_dir):
        for f in files:
            if f.endswith('.npy'):
                allembs.append(np.load(os.path.join(root, f)))
    allembs = np.array(allembs)
    allembs = np.mean(allembs, 0)
    print(f'Embedding shape: {allembs.shape}')
    return allembs

def main(args, dys):
    stats = np.loadtxt(os.path.join(args.mean_std_file_ua, 'global_mean_var.txt'))
    mean = stats[0]
    std = stats[1]
    vc_path = os.path.join(args.model_path_dir, dys, 'vc.pt')
    emb_dir = os.path.join(args.emb_dir, dys)
    vocoder_path = os.path.join(args.vocoder_dir, dys)
    results_dir = os.path.join(args.results_dir, dys)
    cmds = []
    target_cmds = []
    for root, dir, files in os.walk(args.gsc_dir):
        for f in files:
            if f.endswith('.wav'):
                cmds.append(os.path.join(root, f))
    print(len(cmds))
    for root, dir, files in os.walk(os.path.join(args.aty_dir, dys)):
        for f in files:
            if f.endswith('.wav'):
                target_cmds.append(os.path.join(root, f))
    print(len(target_cmds))
    if args.debug:
        cmds = cmds[:2]
        target_cmds = target_cmds[:2]

    allembs = get_avg_emb(emb_dir)
    # loading voice conversion model
    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                       params.layers, params.kernel, params.dropout, params.window_size,
                       params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                       params.beta_min, params.beta_max)
    generator = generator.cuda()
    generator.load_state_dict(torch.load(vc_path))
    generator.eval()
    # loading HiFi-GAN vocoder
    hfg_path = 'checkpts/vocoder/'  # HiFi-GAN path
    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))
    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(vocoder_path + '/g')['generator'])
    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()

    for c in tqdm(cmds):
        tgt_path = target_cmds[0]
        try: 
            inference(args, dys, generator, hifigan_universal, src_path=c, tgt_path=tgt_path, save_path=results_dir, mean=mean, std=std, emb=allembs)
        except:
            print(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/results_allaty')
    parser.add_argument('--model_path_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/logs_dec_aty')
    parser.add_argument('--vocoder_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/hifi-gan/checkpoints')
    parser.add_argument('--aty_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/atypical_data/wav')
    parser.add_argument('--gsc_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/gsc_data/wav')
    parser.add_argument('--phoneme_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/gsc_data/phonemes')
    parser.add_argument('--mean_std_file', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/gsc_data/')
    parser.add_argument('--mean_std_file_ua', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/atypical_data/')
    parser.add_argument('--phoneme_uaspeech', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/atypical_data/')
    parser.add_argument('--emb_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/atypical_data/embeds')
    parser.add_argument('--slow_ratio', type=float, default=1.2)
    parser.add_argument('--fast_ratio', type=float, default=0.8)
    parser.add_argument('--dys', type=str, default='0005')
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    main(args, args.dys)
    # alldys = ['0005','0006','0007','0008','0009','0010','0011','0012','0013','0014','0015','0017','0018','0019','0020']
    # for dys in alldys:
    #     main(args, dys)