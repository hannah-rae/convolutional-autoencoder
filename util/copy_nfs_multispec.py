from subprocess import call, check_call
import csv
import sys
from time import sleep

# This should be copied to then run from an MSL server
# Then you copy the created directory to your local machine or a data directory

data_dir = '/molokini_raid/MSL/data/surface/processed/images/web/full/SURFACE/'
def get_raw_path(instrument, sol, name):
    return data_dir + '%s/%s/%s' % (
        instrument,
        'sol' + str(sol).zfill(4),
        name.strip('\"') + '.png'
    )

csv_fn = "mcam_Lall_Rall_udrs.csv"
dir_name = '/home/hannah/data/' + csv_fn[:-4]
call(['mkdir', dir_name])
with open(csv_fn, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        name, sol, instrument, img_type, filter_used, width, height, filters_right, filters_left, sequence_id, udr_tracking_id, tracking_id = row
        if name == 'name': continue
        call(['mkdir', dir_name + '/sequence_id_' + sequence_id])
        call(['cp',
            get_raw_path(instrument, sol, name),
            dir_name + '/sequence_id_' + sequence_id])
        call(['mv', 
            dir_name + '/sequence_id_' + sequence_id + '/' + name + '.png',
            dir_name + '/sequence_id_' + sequence_id + '/' + name + '_filter' + filter_used + '_sol' + str(sol).zfill(4)])