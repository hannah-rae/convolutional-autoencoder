from glob import glob
from subprocess import call, check_call

data_dir = '/home/hannah/data/mcam_Lall_Rall_udrs'
review_dir = '/home/hannah/data/mcam_multispec_seqs_to_review'
call(['mkdir', review_dir])

for seq_dir in glob(data_dir + '/*'):
	if len(glob(seq_dir + '/*')) != 14:
		print "weirdo found: len is %d" % len(glob(seq_dir + '/*'))
		call(['cp', '-r', seq_dir, review_dir])
