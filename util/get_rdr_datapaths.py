from glob import glob
import numpy as np
import csv

seq_list = '/Users/hannahrae/data/mcam_Lall_Rall_seqids.csv'
expert_list = '/Users/hannahrae/data/mcam_expert_labels.csv'
rdr_list = '/Users/hannahrae/data/mcam_rdr_thumbs_all.csv'

train_list = '/Users/hannahrae/data/mcamrdr_6f_train.csv'
test_list = '/Users/hannahrae/data/mcamrdr_6f_test.csv'

train = []
test = []

# Get a list of expert-selected sequences
with open(expert_list, 'rb') as f:
    reader = csv.reader(f)
    expert_seqids = list(reader)
# flatten the list
#expert_seqids = [item for sublist in expert_seqids for item in sublist]

# Get a list of all sequences
with open(seq_list, 'rb') as f:
    reader = csv.reader(f)
    all_seqids = list(reader)
# flatten the list
all_seqids = [item for sublist in all_seqids for item in sublist]

with open(rdr_list, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        name, path = row
        # sol = name[:4]
        ins = name[4:6]
        if len(name) == 23:
            #print 'short name!'
            seqid = 'mcam' + name[6:10].zfill(5)
        else:
            seqid = 'mcam' + name[7:12]
        #print seqid
        # seqline = name[12:15]
        if seqid in all_seqids and [seqid, ins] not in expert_seqids:
            train.append(path)
        elif seqid in all_seqids and [seqid, ins] in expert_seqids:
            test.append(path)

np.savetxt(train_list, train, fmt='%s')
np.savetxt(test_list, test, fmt='%s')