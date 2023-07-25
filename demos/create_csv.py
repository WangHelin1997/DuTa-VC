import os
import csv
import random

rows = []
fields = ['audio_url']
temp = 'https://mostest2023.s3.us-east-2.amazonaws.com/'
for root, dirs, files in os.walk('/data/dean/whl-2022/Speech-Backbones/DiffVC/am_data_mos'):
    for f in files:
        if '_generated_' in f:
            rows.append([temp+f])
random.shuffle(rows)
# name of csv file
filename = "am_demo.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)
    # writing the data rows
    csvwriter.writerows(rows)