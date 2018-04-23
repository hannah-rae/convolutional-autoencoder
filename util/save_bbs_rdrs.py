# Read the labels and save identified bounding boxes as separate images
from glob import glob
import cv2
import csv
import numpy as np

label_dir = '/home/hannah/data/mcam_DW_bbox_labels'
data_dir = '/home/hannah/data/mcamrdrs/expert'
out_dir = '/home/hannah/data/mcamrdrs/expert/cropped'

# Create a dictionary mapping RDR filename components to bounding boxes
label_map = {}
for label in glob(label_dir+'/*'):
    with open(label, 'rb') as label_file:
        data = label_file.readlines()
        data = [d.rstrip() for d in data]
        # Get the sequence, sol, instrument, and maybe identifier from the filename
        info = label.split('/')[-1].split('_')
        seq = info[0]
        sol = int(info[1][3:]) # take out the prefix 'sol' and zero padding
        if len(info) == 3:
            identifier = info[2][:-4] # do not include file extension
            instrument = None
        else:
            identifier = None
            if 'McamR' in info[2]:
                instrument = 'MR'
            elif 'McamL' in info[2]:
                instrument = 'ML'
        boxes = data[1:]
        label_map[(seq, sol, instrument, identifier)] = boxes

for img_name in glob(data_dir + '/*.npy'):
    # Extract key info from file name
    info = img_name.split('/')[-1].split('_')
    seq = info[0]
    sol = int(info[1])
    ins = info[2]
    if len(info) > 4:
        identifier = info[4][:-4]
        # Get the corresponding bounding boxes
        bboxes = label_map[(seq, sol, None, identifier)]
        label_map.pop((seq, sol, None, identifier))
    else:
        identifier = None
        # Get the corresponding bounding boxes
        bboxes = label_map[(seq, sol, ins, identifier)]
        label_map.pop((seq, sol, ins, identifier))
    print bboxes
    # Load the image
    img = np.load(img_name)
    # Save part of image subtended by each bounding box
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.split()
        x1 = int(x1)-1 # cv2 zero-indexes but boxes were 1-indexed
        y1 = int(y1)-1 # cv2 zero-indexes but boxes were 1-indexed
        x2 = int(x2)-1 # cv2 zero-indexes but boxes were 1-indexed
        y2 = int(y2)-1 # cv2 zero-indexes but boxes were 1-indexed
        crop = img[y1:y1+64, x1:x1+64, :]
        if identifier != None:
            np.save(out_dir + '/' + '%s_%d_%s_%s_%d.npy' % (seq, sol, ins, identifier, i), crop)
        else:
            np.save(out_dir + '/' + '%s_%d_%s_%d.npy' % (seq, sol, ins, i), crop)

print "Leftover images:"
print label_map