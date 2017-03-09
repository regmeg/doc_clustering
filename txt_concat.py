import os
import sys
import subprocess


#indir = '/home/user/projects/data_mining/clustering_cw/gap-html/gap-html/gap_2X5KAAAAYAAJ/text'
indir = '/home/user/projects/data_mining/clustering_cw/gap-html/gap-html'
outdir = '/home/user/projects/data_mining/clustering_cw/book_texts'
#outfile =  '/home/user/projects/data_mining/clustering_cw/book_texts/gap_2X5KAAAAYAAJ.txt'


for dirs in os.listdir(indir):
		dir_path = os.path.join(indir, dirs)
		txt_path = dir_path +'/text'
		for root, drs, filenames in os.walk(txt_path):
			filenames.sort()
			files = filenames
		with open(outdir + "/" + dirs + ".txt", 'w') as outfile:
			for fname in files:	
				with open(txt_path + "/" + fname) as infile:
					outfile.write(infile.read())
