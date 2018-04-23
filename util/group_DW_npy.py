from glob import glob
import os
import csv

data_dir = '/home/hannah/data/mcam_RGB_DW_picks_group_subject'
new_dir = '/home/hannah/data/mcam_multispec_DW_subject/'
map_seqs = '/home/hannah/data/seq_id_map.csv'

# Map right seqids to wrong seqids 
seq_map = {}
with open(map_seqs, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        wrong_id, right_id = row
        seq_map[right_id] = wrong_id

# Copy the multispectral versions of the RGB images to appropriate directory
for subject in glob(data_dir+'/*'):
    for img in glob(subject + '/*'):
        name = img.split('/')[-1]
        seqid = seq_map[name.split('_')[0]]
        sol = name.split('_')[1]
        if 'McamR' in name:
            eye = 'R'
        else:
            eye = 'L'
        new_name = 'sequence_id_' + seqid + '_'+ eye + '*_' + sol

        if 'drillholes' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'drillholes')
        elif 'DRT' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'drt')
        elif 'exposedrock' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'exposedrock')
        elif 'float' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'float')
        elif 'fracturedrocks' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'fracturedrocks')
        elif 'meteorites' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'meteorites')
        elif 'tailingsorpiles' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'tailingsorpiles')
        elif 'veins' in subject:
            os.system('cp ' + new_dir + new_name + '* ' + new_dir + 'veins')