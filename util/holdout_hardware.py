import csv
from glob import glob
from subprocess import call

train_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/train'
hold_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/holdout_hardware'
check_file = '/home/hannah/data/mcam_Lall_Rall_hardware.txt'
map_seqs = '/home/hannah/data/seq_id_map.csv'

# Map right seqids to wrong seqids for converting later
seq_map = {}
with open(map_seqs, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        wrong_id, right_id = row
        seq_map[right_id] = wrong_id

# Create list of sequences in hardware set
hw = set([]) # unordered list of unique images
for line in open(check_file, 'rb'):
    seq = line.split('_')[0]
    if 'McamR' in line.split('_')[2]:
        eye = 'R'
    else:
        eye = 'L'
    seq = seq_map[seq]
    prefix = 'sequence_id_' + seq + '_' + eye
    hw.add(prefix)

hw = list(hw)

# Go through training set and move images of hardware
for example in glob(train_dir + '/*.npy'):
    prefix = example.split('_sol')[0].split('/')[-1][:-1] # we just want the sequence id and eye/num
    #print prefix
    if prefix in hw:
        print 'moving %s to out of train dir' % example
        call(['mv', example, hold_dir])