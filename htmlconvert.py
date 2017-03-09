import os
import sys
import subprocess
import html2text

#indir = '/home/user/projects/data_mining/clustering_cw/gap-html/gap-html'
indir = '/home/user/projects/data_mining/clustering_cw/testing'
#create folders
#for dirs in os.listdir(indir):
#		dir_path = os.path.join(indir, dirs)
#		txt_path = dir_path +'/text'
#		print "creating " + txt_path
#		os.mkdir(txt_path)


#save org stoud
orig_stdout = sys.stdout
#script = "/home/user/projects/data_mining/clustering_cw/htmlconvert.py"
#convert files
i = 0
for root, dirs, filenames in os.walk(indir):
		for f in filenames:
			name, ex = os.path.splitext(f)
			if ex != '.html': break
			i = i + 1
			
total = i
i =- 0

script = '/home/user/projects/data_mining/clustering_cw/html2text.py'

for root, dirs, filenames in os.walk(indir):
		for f in filenames:
			name, ex = os.path.splitext(f)
			if ex != '.html': break
			i = i + 1
			src = os.path.join(root, f)
			trg = os.path.join(root + '/text', name + ".txt")
			#command = "python " + script + " " + src + " > " + trg
			print "running: " + str(i) + " of " + str(total) + " for " + src
			command = "python " + script + " " + src + " > " + trg
			print command
			os.system(command)
			#data = open(src, 'rb').read()
			#try:
			#		from chardet import detect
			#except ImportError:
			#		detect = lambda x: {'encoding': 'utf-8'}
			#encoding = detect(data)['encoding']
			#data = data.decode('utf8')
			#new_file = open(trg,"w")
			#sys.stdout = new_file
			#html2text.wrapwrite(html2text.html2text(data, ''))
			#new_file.close()
			#sys.stdout = orig_stdout

