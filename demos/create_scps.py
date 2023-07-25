import os
import random
import shutil

datapath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/mfa_data'
textpath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/mfa_data'
savepath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/mfa_asrdata_removesil'
augdatapath = '/data/dean/whl-2022/Speech-Backbones/DiffVC/results_allaugdata_removesil'

alldys = ['F02','F03','F04','F05','M01','M04','M05','M07','M08','M09','M10','M11','M12','M14','M16']

def train_ctrl():
    os.makedirs(os.path.join(savepath, 'ctrls', 'train'), exist_ok=True)
    wav_scp_train = open(os.path.join(savepath, 'ctrls', 'train', 'wav.scp'), 'w')
    text_train = open(os.path.join(savepath, 'ctrls', 'train', 'text'), 'w')
    utt2spk_train = open(os.path.join(savepath, 'ctrls', 'train', 'utt2spk'), 'w')
    for root, dirs, files in os.walk(os.path.join(datapath)):
        for f in files:
            if f.endswith('.wav') and f.split('_')[0].startswith('C'):
                wav_scp_train.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                wav_scp_train.write('\n')
                utt2spk_train.write(f.split('.wav')[0] + " " + f.split('_')[0])
                utt2spk_train.write('\n')
                with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                    t = fi.read().replace('\n', '').upper()
                text_train.write(f.split('.wav')[0] + " " + t)
                text_train.write('\n')
    wav_scp_train.close()
    text_train.close()
    utt2spk_train.close()

def train_ctrl_valid():
    os.makedirs(os.path.join(savepath, 'ctrls', 'valid'), exist_ok=True)
    wav_scp_train = open(os.path.join(savepath, 'ctrls', 'valid', 'wav.scp'), 'w')
    text_train = open(os.path.join(savepath, 'ctrls', 'valid', 'text'), 'w')
    utt2spk_train = open(os.path.join(savepath, 'ctrls', 'valid', 'utt2spk'), 'w')
    cmds = []
    for root, dirs, files in os.walk(os.path.join(datapath)):
        for f in files:
            if f.endswith('.wav') and not f.split('_')[0].startswith('C'):
                cmds.append([f, root])
    random.shuffle(cmds)
    cmds = cmds[:800]
    for c in cmds:
        f = c[0]
        root = c[1]
        wav_scp_train.write(f.split('.wav')[0] + " " + os.path.join(root, f))
        wav_scp_train.write('\n')
        utt2spk_train.write(f.split('.wav')[0] + " " + 'dummy')
        utt2spk_train.write('\n')
        with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
            t = fi.read().replace('\n', '').upper()
        text_train.write(f.split('.wav')[0] + " " + t)
        text_train.write('\n')
    wav_scp_train.close()
    text_train.close()
    utt2spk_train.close()

def train_ctrl_test():
    os.makedirs(os.path.join(savepath, 'ctrls', 'test'), exist_ok=True)
    for dys in alldys:
        shutil.copytree(os.path.join(savepath, dys, 'test'), os.path.join(savepath, 'ctrls', 'test_'+dys))

def train_dys():
    for dys in alldys:
        os.makedirs(os.path.join(savepath, dys, 'train'), exist_ok=True)
        os.makedirs(os.path.join(savepath, dys, 'test'), exist_ok=True)
        # split train, valid and test
        wav_scp_train = open(os.path.join(savepath, dys, 'train', 'wav.scp'), 'w')
        text_train = open(os.path.join(savepath, dys, 'train', 'text'), 'w')
        utt2spk_train = open(os.path.join(savepath, dys, 'train', 'utt2spk'), 'w')
        wav_scp_test = open(os.path.join(savepath, dys, 'test', 'wav.scp'), 'w')
        text_test = open(os.path.join(savepath, dys, 'test', 'text'), 'w')
        utt2spk_test = open(os.path.join(savepath, dys, 'test', 'utt2spk'), 'w')
        for root, dirs, files in os.walk(os.path.join(datapath, dys)):
            for f in files:
                if f.endswith('.wav'):
                    if f.split('_')[1] != 'B2':
                        wav_scp_train.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                        wav_scp_train.write('\n')
                        utt2spk_train.write(f.split('.wav')[0] + " " + dys)
                        utt2spk_train.write('\n')
                        with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                            t = fi.read().replace('\n', '').upper()
                        text_train.write(f.split('.wav')[0] + " " + t)
                        text_train.write('\n')
                    else:
                        wav_scp_test.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                        wav_scp_test.write('\n')
                        utt2spk_test.write(f.split('.wav')[0] + " " + dys)
                        utt2spk_test.write('\n')
                        with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                            t = fi.read().replace('\n', '').upper()
                        text_test.write(f.split('.wav')[0] + " " + t)
                        text_test.write('\n')
        for root, dirs, files in os.walk(os.path.join(datapath)):
            for f in files:
                if f.endswith('.wav') and f.split('_')[0].startswith('C'):
                    wav_scp_train.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                    wav_scp_train.write('\n')
                    utt2spk_train.write(f.split('.wav')[0] + " " + f.split('_')[0])
                    utt2spk_train.write('\n')
                    with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                        t = fi.read().replace('\n', '').upper()
                    text_train.write(f.split('.wav')[0] + " " + t)
                    text_train.write('\n')

        wav_scp_train.close()
        wav_scp_test.close()
        text_train.close()
        text_test.close()
        utt2spk_train.close()
        utt2spk_test.close()

def train_dys_aug():
    for dys in alldys:
        os.makedirs(os.path.join(savepath, dys+'_aug', 'train'), exist_ok=True)
        os.makedirs(os.path.join(savepath, dys+'_aug', 'test'), exist_ok=True)
        # split train, valid and test
        wav_scp_train = open(os.path.join(savepath, dys+'_aug', 'train', 'wav.scp'), 'w')
        text_train = open(os.path.join(savepath, dys+'_aug', 'train', 'text'), 'w')
        utt2spk_train = open(os.path.join(savepath, dys+'_aug', 'train', 'utt2spk'), 'w')
        wav_scp_test = open(os.path.join(savepath, dys+'_aug', 'test', 'wav.scp'), 'w')
        text_test = open(os.path.join(savepath, dys+'_aug', 'test', 'text'), 'w')
        utt2spk_test = open(os.path.join(savepath, dys+'_aug', 'test', 'utt2spk'), 'w')
        for root, dirs, files in os.walk(os.path.join(datapath, dys)):
            for f in files:
                if f.endswith('.wav'):
                    if f.split('_')[1] != 'B2':
                        wav_scp_train.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                        wav_scp_train.write('\n')
                        utt2spk_train.write(f.split('.wav')[0] + " " + 'dummy')
                        utt2spk_train.write('\n')
                        with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                            t = fi.read().replace('\n', '').upper()
                        text_train.write(f.split('.wav')[0] + " " + t)
                        text_train.write('\n')
                    else:
                        wav_scp_test.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                        wav_scp_test.write('\n')
                        utt2spk_test.write(f.split('.wav')[0] + " " + 'dummy')
                        utt2spk_test.write('\n')
                        with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                            t = fi.read().replace('\n', '').upper()
                        text_test.write(f.split('.wav')[0] + " " + t)
                        text_test.write('\n')

        for root, dirs, files in os.walk(os.path.join(augdatapath, dys)):
            for f in files:
                if f.endswith('.wav'):
                    wav_scp_train.write(f.split('.wav')[0]+'_aug' + " " + os.path.join(root, f))
                    wav_scp_train.write('\n')
                    utt2spk_train.write(f.split('.wav')[0]+'_aug' + " " + 'dummy')
                    utt2spk_train.write('\n')
                    with open(os.path.join(datapath, f.split('_')[0], f.replace('.wav', '.lab'))) as fi:
                        t = fi.read().replace('\n', '').upper()
                    text_train.write(f.split('.wav')[0]+'_aug' + " " + t)
                    text_train.write('\n')

        for root, dirs, files in os.walk(os.path.join(datapath)):
            for f in files:
                if f.endswith('.wav') and f.split('_')[0].startswith('C'):
                    wav_scp_train.write(f.split('.wav')[0] + " " + os.path.join(root, f))
                    wav_scp_train.write('\n')
                    utt2spk_train.write(f.split('.wav')[0] + " " + 'dummy')
                    utt2spk_train.write('\n')
                    with open(os.path.join(root, f.replace('.wav', '.lab'))) as fi:
                        t = fi.read().replace('\n', '').upper()
                    text_train.write(f.split('.wav')[0] + " " + t)
                    text_train.write('\n')

        wav_scp_train.close()
        wav_scp_test.close()
        text_train.close()
        text_test.close()
        utt2spk_train.close()
        utt2spk_test.close()

# egs2/TEMPLATE/asr1/setup.sh egs2/uaspeech_ctrl/asr1
# scp -r egs2/uaspeech/asr1/asr.sh egs2/uaspeech_ctrl/asr1/asr.sh
# scp -r egs2/uaspeech/asr1/run.sh egs2/uaspeech_ctrl/asr1/run.sh
## modify run.sh

#
# scp -r /data/dean/whl-2022/Speech-Backbones/DiffVC/mfa_asrdata/F02_aug /data/dean/whl-2022/espnet/egs2/uaspeech_F02_aug/asr1/data

# cd /data/dean/whl-2022/espnet/egs2/wsj/asr1/
# utils/fix_data_dir.sh /data/dean/whl-2022/espnet/egs2/uaspeech_F02_aug/asr1/data/train
# utils/spk2utt_to_utt2spk.pl data/train/spk2utt > data/train/utt2spk

# cd /data/dean/whl-2022/espnet/egs2/uaspeech_F02_aug/asr1

#./asr.sh --stage 2 --ngpu 1 --train_set train --valid_set test --test_sets "test" --lm_train_text "data/train/text"

train_dys()
train_dys_aug()
train_ctrl()
train_ctrl_valid()
train_ctrl_test()