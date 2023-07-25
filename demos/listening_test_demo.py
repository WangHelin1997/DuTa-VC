import os
import librosa
import soundfile as sf
import random
import numpy as np
from librosa.core import load
import csv

source_path = '/data/dean/whl-2022/Speech-Backbones/DiffVC/mfa_data'
fake_path = '/data/dean/whl-2022/Speech-Backbones/DiffVC/results_allaugdata'
save_path = '/data/dean/whl-2022/Speech-Backbones/DiffVC/listening_test'
sample_number = 15
count = 1
total_sample = 3
alldys = ['F02','F03','F04','F05','M01','M04','M05','M07','M08','M09','M10','M11','M12','M14','M16']
fields1 = ['filename', 'speaker', 'severity', 'type of dysarthric', 'others', 'transcriptions']
fields2 = ['filename', 'severity', 'type of dysarthric', 'others', 'transcriptions']
rows1 = []
rows2 = []
os.makedirs(save_path, exist_ok=True)
allcmds = []
for i in range(total_sample):
    for dys in alldys:
        source_cmds = []
        for root, dirs, files in os.walk(os.path.join(source_path, dys)):
            for f in files:
                if f.endswith('.wav'):
                    source_cmds.append(os.path.join(root, f))
        random.shuffle(source_cmds)

        cmds = source_cmds[:sample_number]
        text = ''
        audio = np.zeros(10)
        for c in cmds:
            wav, _ = load(c, sr=22050)
            audio = np.append(audio, wav, 0)
            audio = np.append(audio, np.zeros(22050), 0)
            with open(os.path.join(source_path, dys, c.split('/')[-1].replace('.wav', '.lab'))) as fi:
                tt = fi.readline()
            text += tt + ' '
        print(audio.shape)
        allcmds.append([dys, audio, text])

        ge_cmds = []
        text = ''
        for root, dirs, files in os.walk(os.path.join(fake_path, dys)):
            for f in files:
                if f.endswith('.wav'):
                    ge_cmds.append(os.path.join(root, f))
        random.shuffle(ge_cmds)
        cmds = ge_cmds[:sample_number-3]
        ge_cmds = []
        for root, dirs, files in os.walk(os.path.join(source_path, dys)):
            for f in files:
                if f.endswith('.wav'):
                    ge_cmds.append(os.path.join(root, f))
        random.shuffle(ge_cmds)
        cmds = cmds + ge_cmds[:3]
        random.shuffle(cmds)
        audio = np.zeros(10)
        for c in cmds:
            wav, _ = load(c, sr=22050)
            audio = np.append(audio, wav, 0)
            audio = np.append(audio, np.zeros(22050), 0)
            with open(os.path.join(source_path, c.split('/')[-1].split('_')[0], c.split('/')[-1].replace('.wav', '.lab'))) as fi:
                tt = fi.readline()
            text += tt + ' '
        print(audio.shape)
        allcmds.append([dys+'_syn', audio, text])

random.shuffle(allcmds)
os.makedirs(os.path.join(save_path, 'audios'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'transcriptions'), exist_ok=True)
for c in allcmds:
    savename = '{:03}'.format(count) + '.wav'
    rows1.append([savename, c[0], '', '', '', c[2]])
    rows2.append([savename, '', '', '', c[2]])
    sf.write(os.path.join(save_path, 'audios', savename), c[1], 22050, 'PCM_24')
    with open(os.path.join(save_path, 'transcriptions', savename.replace('.wav', '.txt')), 'w') as fi:
        fi.write(c[2])
    count += 1

filename1 = os.path.join(save_path, "listening_test_spk.csv")
filename2 = os.path.join(save_path, "listening_test.csv")

with open(filename1, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields1)
    csvwriter.writerows(rows1)

with open(filename2, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields2)
    csvwriter.writerows(rows2)