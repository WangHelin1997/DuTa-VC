import os
from tqdm import tqdm
import shutil
import random

resultdir = '/data/dean/whl-2022/Speech-Backbones/DiffVC/results_allaugdata_removesil'
audiodir = '/data/dean/whl-2022/Speech-Backbones/DiffVC/mfa_data'
savedir = '/data/dean/whl-2022/Speech-Backbones/DiffVC/am_data_mos'

# alldys = ['F02','F03','F04','F05','M01','M04','M05','M07','M08','M09','M10','M11','M12','M14','M16']
alldys = ['M08','M10','M05','M11','M04','M12']

for dys in alldys:
    cmds = []
    for root, dirs, files in os.walk(os.path.join(resultdir, dys)):
        for f in files:
            if f.endswith('.wav') and f.split('_')[1] == 'B2':
                cmds.append([root, f])
    print(len(cmds))
    random.shuffle(cmds)
    cmds = cmds[:50]
    t_cmds = []
    for root, dirs, files in os.walk(os.path.join(audiodir, dys)):
        for f in files:
            if f.endswith('.wav') and f.split('_')[1] != 'B2':
                t_cmds.append(os.path.join(root, f))
    print(len(t_cmds))
    random.shuffle(t_cmds)
    t_cmds = t_cmds[:50]
    datas = []
    for i, c in enumerate(cmds):
        gt = os.path.join(audiodir, dys, dys+'_'+c[1].split('_',1)[-1])
        if os.path.exists(gt):
            generated = os.path.join(c[0], c[1])
            source = os.path.join(audiodir, c[1].split('_')[0], c[1])
            target = t_cmds[i]
            datas.append([source, target, generated, gt])
    print(len(datas))
    os.makedirs(os.path.join(savedir, dys), exist_ok=True)
    count = 1
    for d in datas:
        shutil.copyfile(d[0], os.path.join(savedir, dys, str(count)+'_'+dys+'_source_'+d[0].split('/')[-1]))
        shutil.copyfile(d[1], os.path.join(savedir, dys, str(count)+'_'+dys + '_target_' + d[1].split('/')[-1]))
        shutil.copyfile(d[2], os.path.join(savedir, dys, str(count)+'_'+dys + '_generated_' + d[2].split('/')[-1]))
        # shutil.copyfile(d[3], os.path.join(savedir, dys, str(count) + '_gt_' + d[3].split('/')[-1]))
        count += 1
