from glob import glob
from subprocess import call, check_call

data_dir = '/home/hannah/data/mcam_Lall_Rall_udrs'

# Find images that need to be deleted since they aren't in a directory
for seq_dir in glob(data_dir + '/*'):
	for img in glob(seq_dir + '/Mcam*'):
		print "deleting ", img
		call(['rm', img])