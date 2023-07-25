import os
import csv
from tqdm import tqdm

csvpath = '/scratch4/lmorove1/hwang258/data/atypicalspeech/atypicalspeech/excel_out_slurp_metadata.csv'
savepath = '/scratch4/lmorove1/hwang258/data/atypicalspeech/atypicalspeech/excel_out_slurp_metadata_named.csv'
lines = []
# opening the CSV file 
with open(csvpath, mode ='r')as file:
      csvFile = csv.reader(file) 
      for line in csvFile: 
        lines.append(line)
fields = lines[0]
lines = lines[1:]
dicts = {}
savelines = []
for line in tqdm(lines):
    name = line[2]
    if name not in dicts.keys():
        dicts[name] = 1
    else:
        dicts[name] += 1
    savename = 'XXXX_script'+name+'_line000'+str(dicts[name])+'.wav'
    line[2] = savename
    savelines.append(line)

with open(savepath, 'w') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields) 
    csvwriter.writerows(savelines)
