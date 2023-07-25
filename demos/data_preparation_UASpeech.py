import argparse
import json
import os
import numpy as np
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
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
import random
from tqdm import tqdm
import soundfile as sf
import multiprocessing
import shutil

def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path, spk_encoder):
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

def load_model():
    # loading voice conversion model
    vc_path = 'checkpts/vc/vc_libritts_wodyn.pt'  # path to voice conversion model

    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                       params.layers, params.kernel, params.dropout, params.window_size,
                       params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                       params.beta_min, params.beta_max)
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_path))
    else:
        generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
    generator.eval()

    print(f'Number of parameters: {generator.nparams}')

    # loading HiFi-GAN vocoder
    hfg_path = 'checkpts/vocoder/'  # HiFi-GAN path

    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))

    if use_gpu:
        hifigan_universal = HiFiGAN(h).cuda()
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
    else:
        hifigan_universal = HiFiGAN(h)
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator', map_location='cpu')['generator'])

    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()

    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
    if use_gpu:
        spk_encoder.load_model(enc_model_fpath, device="cuda")
    else:
        spk_encoder.load_model(enc_model_fpath, device="cpu")
    return generator, hifigan_universal, spk_encoder

def inference(args, src_path, tgt_path, save_path, s_id, generator, hifigan_universal, spk_encoder):
    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    os.makedirs(save_path, exist_ok=True)
    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    if use_gpu:
        mel_source = mel_source.cuda()
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    if use_gpu:
        mel_source_lengths = mel_source_lengths.cuda()

    mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        mel_target = mel_target.cuda()
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
    if use_gpu:
        mel_target_lengths = mel_target_lengths.cuda()

    embed_target = torch.from_numpy(get_embed(tgt_path, spk_encoder)).float().unsqueeze(0)
    if use_gpu:
        embed_target = embed_target.cuda()

    # performing voice conversion
    mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
                                          n_timesteps=30, mode='ml')
    mel_synth_np = mel_.cpu().detach().squeeze().numpy()
    mel_source_np = mel_.cpu().detach().squeeze().numpy()
    mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
    tag = os.path.basename(tgt_path).split('.wav')[0]
    if use_gpu:
        mel = mel.cuda()
    if args.debug:
        # source utterance (vocoded)
        with torch.no_grad():
            audio = hifigan_universal.forward(mel_source).cpu().reshape(1, -1)
        torchaudio.save(os.path.join(save_path, tag+'_'+s_id+'_source.wav'), audio, 22050)

        # reference utterance (vocoded)
        with torch.no_grad():
            audio = hifigan_universal.forward(mel_target).cpu().reshape(1, -1)
        torchaudio.save(os.path.join(save_path, tag+'_'+s_id+'_target.wav'), audio, 22050)
    # converted speech
    with torch.no_grad():
        audio = hifigan_universal.forward(mel).cpu().squeeze().reshape(1, -1)
    torchaudio.save(os.path.join(save_path, tag+'_'+s_id+'_converted.wav'), audio, 22050)

def test(args):
    train_commands = {}
    test_commands = {}
    commands = {}
    count_utterances = 0
    for root, dirs, files in os.walk(args.source_wav_dir):
        for f in files:
            if f.endswith('M5.wav'):
                if f.startswith('C') or f.startswith('M') or f.startswith('F'):
                    id = f.split('_')[0]
                    # name = f.split('_')[2]
                    name = f.rsplit("_", 1)[0].split('.wav')[0].split('_', 1)[-1]
                    if id not in commands.keys():
                        commands[id] = {}
                    if name not in commands[id].keys():
                        commands[id][name] = [f.split('.wav')[0], os.path.join(root, f)]
                        count_utterances += 1
    print(f"Total ids: {len(commands)}")
    print(f"Total utterances: {count_utterances}")
    for k in list(commands.keys()):
        if k in args.test_list:
            test_commands[k] = commands[k]
        else:
            train_commands[k] = commands[k]
    print(f"Total train ids: {len(train_commands)}, test ids: {len(test_commands)}")
    generator, hifigan_universal, spk_encoder = load_model()

    ids = list(train_commands.keys())
    dys_ids = []
    ctrl_ids = []
    run_commands = []
    missings = 0
    for id in ids:
        if id.startswith('C'):
            ctrl_ids.append(id)
        else:
            dys_ids.append(id)
    if args.debug:
        ids = ids[:5]
    for id in ids:
        names = list(train_commands[id].keys())
        # print(len(names))
        if args.debug:
            names = names[:2]
        for name in names:
            source_list = random.sample(dys_ids, args.sample_num) if id.startswith('C') else random.sample(ctrl_ids, args.sample_num)
            # source_list = dys_ids if id.startswith('C') else ctrl_ids
            # source_list = source_list * args.sample_num
            for s_id in source_list:
                if name in train_commands[s_id].keys():
                    src_path = train_commands[s_id][name][1]
                    tgt_path = train_commands[id][name][1]
                    save_path = os.path.join(args.save_path, id)
                    # print(src_path, tgt_path, save_path)
                    run_commands.append([src_path, tgt_path, save_path, s_id])
                    # run_commands.append((args, src_path, tgt_path, save_path, s_id, generator, hifigan_universal, spk_encoder))
                else:
                    missings += 1
    print(f'Total commands: {len(run_commands)}, total missing files: {missings}')

    for c in tqdm(run_commands):
        inference(args, c[0], c[1], c[2], c[3], generator, hifigan_universal, spk_encoder)

def segment_one(args, segment_file):
    os.makedirs(os.path.join(args.processed_wav_dir, segment_file[0]), exist_ok=True)
    with open(segment_file[1]) as f:
        segs = f.readlines()
        if args.debug:
            segs = segs[:10]
        for s in tqdm(segs):
            commands = s.split('\n')[0].split(' ')
            if commands[1].endswith('M5'):
                audiopath = os.path.join(args.source_wav_dir, segment_file[0], commands[1]+'.wav')
                if os.path.exists(audiopath):
                    audio, _ = load(audiopath, sr=22050, offset=float(commands[2]), duration=float(commands[3])-float(commands[2]))
                    savename = os.path.join(args.processed_wav_dir, segment_file[0], commands[0]+'.wav')
                    sf.write(savename, audio, 22050, 'PCM_24')
                else:
                    print(f'Missing file: {audiopath}')

def get_duration(args):
    train_segment_files, test_segment_files = [], []
    for root, dirs, files in os.walk(args.segment_dir):
        for f in files:
            if f.endswith('.segments'):
                if f.split('.')[0] in args.test_list:
                    test_segment_files.append([f.split('.')[0], os.path.join(root, f)])
                else:
                    train_segment_files.append([f.split('.')[0], os.path.join(root, f)])

    print(f'Total train segment files: {len(train_segment_files)}, test segment files: {len(test_segment_files)}')
    train_duration = 0.
    test_duration = 0.
    for segment_file in train_segment_files:
        with open(segment_file[1]) as f:
            segs = f.readlines()
            for s in segs:
                commands = s.split('\n')[0].split(' ')
                if commands[1].endswith('M5'):
                    audiopath = os.path.join(args.source_wav_dir, segment_file[0], commands[1]+'.wav')
                    if os.path.exists(audiopath):
                        train_duration += float(commands[3])-float(commands[2])
    for segment_file in test_segment_files:
        with open(segment_file[1]) as f:
            segs = f.readlines()
            for s in segs:
                commands = s.split('\n')[0].split(' ')
                if commands[1].endswith('M5'):
                    audiopath = os.path.join(args.source_wav_dir, segment_file[0], commands[1]+'.wav')
                    if os.path.exists(audiopath):
                        test_duration += float(commands[3])-float(commands[2])
    print(f'Total train duration: {train_duration/3600}, test duration: {test_duration/3600}')

def segment(args):
    segment_files = []
    for root, dirs, files in os.walk(args.segment_dir):
        for f in files:
            if f.endswith('.segments'):
                segment_files.append([f.split('.')[0], os.path.join(root, f)])
    cmds = [(args, s) for s in segment_files]
    print(f'Total segment files: {len(segment_files)}')
    if args.debug:
        cmds = cmds[:2]
    with multiprocessing.Pool(processes=len(segment_files)) as pool:
        pool.starmap(segment_one, cmds)

def data_generator(command, spk_encoder):
    os.makedirs(os.path.join(command[1], 'wavs', command[2]), exist_ok=True)
    os.makedirs(os.path.join(command[1], 'mels', command[2]), exist_ok=True)
    os.makedirs(os.path.join(command[1], 'embeds', command[2]), exist_ok=True)
    mel = get_mel(command[0])
    emb = get_embed(command[0], spk_encoder)
    shutil.copyfile(command[0], os.path.join(command[1], 'wavs', command[2], command[0].split('/')[-1]))
    np.save(os.path.join(command[1], 'mels', command[2], command[0].split('/')[-1].replace('.wav', '.npy')), mel)
    np.save(os.path.join(command[1], 'embeds', command[2], command[0].split('/')[-1].replace('.wav', '.npy')), emb)

def split_train_data(args):
    commands = []
    for root, dirs, files in os.walk(args.processed_wav_dir):
        for f in files:
            if f.endswith('.wav'):
                tag = f.split('_')[0]
                if tag.startswith('C'):
                    wav_path = os.path.join(args.train_dir_c, 'wavs', tag)
                    mel_path = os.path.join(args.train_dir_c, 'mels', tag)
                    embed_path = os.path.join(args.train_dir_c, 'embeds', tag)
                    commands.append([os.path.join(root, f), args.train_dir_c, tag])
                else:
                    wav_path = os.path.join(args.train_dir_d, 'wavs', tag)
                    mel_path = os.path.join(args.train_dir_d, 'mels', tag)
                    embed_path = os.path.join(args.train_dir_d, 'embeds', tag)
                    commands.append([os.path.join(root, f), args.train_dir_d, tag])
    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
    spk_encoder.load_model(enc_model_fpath, device="cpu")
    if args.debug:
        commands = commands[:10]
    for command in tqdm(commands):
        data_generator(command, spk_encoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_wav_dir', type=str,
                        default='/data/dean/whl-2022/UASpeech/ifp-08.ifp.uiuc.edu/protected/UASpeech/audio/noisereduce')
    parser.add_argument('--processed_wav_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/processed_data')
    parser.add_argument('--train_dir_c', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/train_dir_c')
    parser.add_argument('--train_dir_d', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/DiffVC/train_dir_d')
    parser.add_argument('--save_path', type=str, default='/data/dean/whl-2022/Speech-Backbones/DiffVC/synthesized_data')
    parser.add_argument('--segment_dir', type=str,
                        default='/data/dean/whl-2022/Speech-Backbones/uaspeech/s5_segment/local/flist/mlf')
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--mode', type=str, default='a2n')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    # parser.add_argument('--test_list', type=list, default=['F03', 'M05', 'M08', 'CF03', 'CM01', 'CM10'])
    parser.add_argument('--test_list', type=list, default=['M05', 'CM01'])
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    # segment(args)
    # get_duration(args)
    split_train_data(args)