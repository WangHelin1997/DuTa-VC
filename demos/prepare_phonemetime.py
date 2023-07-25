import os
import tgt
import numpy as np
from tqdm import tqdm
import multiprocessing

phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'sil',
                'sp', 'spn']


def process_one(textgrid, savepath):
    try:
        t = tgt.io.read_textgrid(textgrid)
        tp = t.get_tier_by_name('phones')
        tw = t.get_tier_by_name('words')
        allphones = []
        alldurations = []

        words = []
        for i in range(len(tw)):
            if (i == 0 or i == len(tw) - 1) and (
                    tp[i].text == '' or tp[i].text == 'sil' or tp[i].text != 'sp' or tp[i].text == 'spn'):
                continue
            words.append([tw[i].start_time, tw[i].end_time])

        j = 0
        for i in range(len(tp)):
            phoneme = tp[i].text
            start = tp[i].start_time
            end = tp[i].end_time
            if words[j][0] <= start and words[j][1] >= end:
                frame_num = int(np.ceil((end - start) * 22050.0 // 256))
                if phoneme == '' or phoneme == 'sil' or phoneme == 'sp' or phoneme == 'spn':
                    allphones.append(phoneme_list.index('sil'))
                else:
                    allphones.append(phoneme_list.index(phoneme))
                alldurations.append(frame_num)
            if words[j][1] == end:
                j += 1
                if j == len(words):
                    break
        allphones = np.array(allphones)
        alldurations = np.array(alldurations)
        # print(allphones)
        # print(alldurations)
        print(textgrid)
        np.save(os.path.join(savepath, 'ttsphonemes', textgrid.split('/')[-1].replace('.TextGrid', '.npy')), allphones)
        np.save(os.path.join(savepath, 'ttsdurations', textgrid.split('/')[-1].replace('.TextGrid', '.npy')), alldurations)
    except:
        print(f'error:{textgrid}')

def process_files(textgrids):
    for textgrid in tqdm(textgrids):
        process_one(textgrid)

# textgrid = '/data/dean/whl-2022/LibriMix/data/librispeech/text/dev-clean/84/121550/84-121550-0000.TextGrid'
# textgrids = '/data/dean/whl-2022/Speech-Backbones/DiffVC/librispeechData/textgrids'
# savepath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/librispeechData'
textgrids = '/data/dean/whl-2022/Speech-Backbones/TextGrid/LJSpeech'
savepath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/LJSpeechData'
os.makedirs(os.path.join(savepath, 'ttsphonemes'), exist_ok=True)
os.makedirs(os.path.join(savepath, 'ttsdurations'), exist_ok=True)
cmds = []
for root, dirs, files in os.walk(textgrids):
    for f in files:
        if f.endswith('.TextGrid'):
            cmds.append((os.path.join(root, f), savepath))
print(len(cmds))
with multiprocessing.Pool(processes=50) as pool:
    pool.starmap(process_one, cmds)