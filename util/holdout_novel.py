import csv
from glob import glob
from subprocess import call

train_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/train'
hold_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/holdout'
check_dir = '/home/hannah/data/mcam_multispec_DW'

# Create list of sequences in expert set
expert = set([]) # unordered list of unique images
for img in glob(check_dir + '/*'):
    prefix = img.split('_sol')[0].split('/')[-1] # we just want the sequence id and eye/num
    #print prefix
    expert.add(prefix)

# Go through training set and move images that came from expert-selected
# sequences to hold out set (TODO: change name of this holdout dir, misleading)
for example in glob(train_dir + '/*'):
    prefix = example.split('_sol')[0].split('/')[-1] # we just want the sequence id and eye/num
    #print prefix
    if prefix in expert:
        print 'moving %s to out of train dir' % example
        call(['mv', example, hold_dir])