import numpy as np
from idlpy import IDL
import time

train = '/home/hannah/data/mcamrdr_train_seqsol.txt'
expert = '/home/hannah/data/mcamrdr_expert_seqsol.txt'
data_dir = '/home/hannah/data/mcamrdrs'

getfile_command = """img_list = hanika_mcam_get_filelist(%s, 'RAD', sequence='%s', /thumbnails, /ignore_focus_thumbnails, /remove_duplicates, /resolve_multiple_pointings, /resolve_multiple_filters, /ignore_old)"""
readpds_command = """img = mer_readpds('%s')"""

blacklist_seqs = ['mcam00567', 'mcam00047'] # Sequences that are missing data or have weird properties

def get_pyimage_arrays(sol, seq, train):
    # Get an array where each element is a list of files with the same pointing
    print getfile_command % (sol, seq)
    IDL.run(getfile_command % (sol, seq))
    img_list = IDL.img_list
    print img_list

    image_names = set([])
    for img_set in img_list:
        if len(img_set) < 6:
            print "# images = %d, skipping" % len(img_set)
            continue

        filter_count = 0
        for file in img_set:
            # We don't want to use the 0 (Bayer RGB true color) filter
            if 'A0' in file:
                print "Skipping A0"
                continue
            # Read the PDS file into an IDL array
            IDL.run(readpds_command % file)
            py_img = IDL.img

            print 'SHAPE: ', py_img.shape

            # Numpy references image dimensions as Y x X x N,
            # whereas IDL references as N x X x Y. We only keep
            # the relevant Bayer band with the most info for that filter.
            if 'ML' in file:
                ins = 'ML'
                if 'A1' in file:
                    img_0 = py_img[1,:,:]
                    filter_count += 1
                elif 'A2' in file:
                    img_1 = py_img[2,:,:]
                    filter_count += 1
                elif 'A3' in file:
                    img_2 = py_img[0,:,:]
                    filter_count += 1
                elif 'A4' in file:
                    img_3 = py_img[0,:,:]
                    filter_count += 1
                elif 'A5' in file:
                    img_4 = py_img[1,:,:]
                    filter_count += 1
                elif 'A6' in file:
                    img_5 = py_img[1,:,:]
                    filter_count += 1
            elif 'MR' in file:
                ins = 'MR'
                if 'A1' in file:
                    img_0 = py_img[1,:,:]
                    filter_count += 1
                elif 'A2' in file:
                    img_1 = py_img[2,:,:]
                    filter_count += 1
                elif 'A3' in file:
                    img_2 = py_img[0,:,:]
                    filter_count += 1
                elif 'A4' in file:
                    img_3 = py_img[1,:,:]
                    filter_count += 1
                elif 'A5' in file:
                    img_4 = py_img[1,:,:]
                    filter_count += 1
                elif 'A6' in file:
                    img_5 = py_img[1,:,:]
                    filter_count += 1

        # Check that it was actually a complete multispectral image
        if filter_count == 6:
            print "All filters accounted for"
            img_6f = np.ndarray([img_0.shape[0],img_0.shape[1],6])
            img_6f[:,:,0] = img_0
            img_6f[:,:,1] = img_1
            img_6f[:,:,2] = img_2
            img_6f[:,:,3] = img_3
            img_6f[:,:,4] = img_4
            img_6f[:,:,5] = img_5

            version = file.split('R')[-1][0]
            new_name = '%s_%s_%s_%s.npy' % (seq, sol, ins, version)
            if new_name in image_names:
                new_name = '%s_%s_%s_%s_%d.npy' % (seq, sol, ins, version, int(time.time()*1000))
            else:
                image_names.add(new_name)
            if train:
                np.save(data_dir + '/train/' + new_name, img_6f)
            else:
                np.save(data_dir + '/expert/' + new_name, img_6f)

with open(train, 'rb') as train_file:
    for line in train_file:
        sol, seqid = line.split(',')
        print sol
        print seqid
        if seqid.rstrip() in blacklist_seqs:
            print "skipping blacklisted sequence"
            continue
        get_pyimage_arrays(sol, seqid.rstrip(), train=True)

with open(expert, 'rb') as expert_file:
    for line in expert_file:
        sol, seqid = line.split(',')
        get_pyimage_arrays(sol, seqid.rstrip(), train=False)
