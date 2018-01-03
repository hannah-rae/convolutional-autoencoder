from glob import glob
from subprocess import call, check_call

data_dir = '/home/hannah/data/mcam_Lall_Rall_udrs'

for seq_dir in glob(data_dir + '/*'):
	if len(glob(seq_dir + '/*')) != 14:
		# ignore these since they are manually handled
		continue
	call(['mkdir', seq_dir + '/L1'])
	call(['mkdir', seq_dir + '/R1'])
	for img in glob(seq_dir + '/*'):
		img_name = img.split('/')[-1]
		if img_name.startswith('McamL'):
			call(['mv', img, seq_dir + '/L1'])
		elif img_name.startswith('McamR'):
			call(['mv', img, seq_dir + '/R1'])
	assert len(glob(seq_dir + '/L1/*')) == 7, 'insufficient left images'
	assert len(glob(seq_dir + '/R1/*')) == 7, 'insufficient right images'