from glob import glob
import csv
import numpy as np

old_test = '/Users/hannahrae/data/mcam_expert_labels.csv'
old_train = '/Users/hannahrae/data/mcam_rdr_thumbs_all.csv'

new_test = '/Users/hannahrae/data/mcamrdr_expert_seqsol.txt'
new_train = '/Users/hannahrae/data/mcamrdr_train_seqsol.txt'

expert_list = set([])
with open(old_test, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for seqid, ins in reader:
        expert_list.add(seqid)
expert_list = list(expert_list)

train = set([])
expert = set([])
with open(old_train, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for name, path in reader:
        sol = int(name[:4])
        if sol > 1666:
            continue
        #print sol
        if len(name) == 23:
            #print 'short name!'
            seqid = 'mcam' + name[6:10].zfill(5)
        else:
            seqid = 'mcam' + name[7:12]
        if seqid in expert_list:
            expert.add((sol,seqid))
        else:
            train.add((sol,seqid))

train = [str(x[0]) +','+ str(x[1]) for x in list(train)]
expert = [str(x[0]) +','+ str(x[1]) for x in list(expert)]

np.savetxt(new_test, expert, delimiter='\n', fmt='%s')
np.savetxt(new_train, train, delimiter='\n', fmt='%s')