import SimpleITK as sitk
import os, glob
import json
import numpy as np
keyword = 'train'

dictout = {keyword:[]}
filenamelist = []
for idx, filename in enumerate (sorted(glob.glob("./*.mhd"), key=os.path.basename)):
	filenamelist.append(filename)

for i in range (0,len(filenamelist)):
	for j in range (0, len(filenamelist)):
		if (i!=j):
			smalldict = {}
			print(filenamelist[i][2:], filenamelist[j][2:])
			smalldict['source'] = filenamelist[i][2:]
			smalldict['target'] = filenamelist[j][2:]
			dictout['train'].append(smalldict)
	# img = sitk.GetArrayFromImage(sitk.ReadImage(filename))

savefilename = './train'+ '.json'
with open(savefilename, 'w') as fp:
	json.dump(dictout, fp)
