from glob import glob
import numpy as np
import os

label_dir = '/Users/hannahrae/src/BBox-Label-Tool/Labels/003'
label_csv = '/Users/hannahrae/data/mcam_expert_labels_.csv'

labels = []
for lbl in glob(label_dir + '/*'):
    seqid = lbl.split('/')[-1].split('_')[0]
    ins = ''
    if 'McamL' in lbl.split('/')[-1].split('_')[2]:
        ins = 'ML'
    else:
        ins = 'MR'
    s = os.path.getsize(lbl) # check that it had a bounding box
    if s > 2 and [seqid, ins] not in labels:
        labels.append([seqid, ins])

labels = np.asarray(labels)
with open(label_csv, 'wb'):
    np.savetxt(label_csv, labels, delimiter=',', fmt='%s')