import csv

file = open("labels-map-proj-v3.txt", "r")
data = list(csv.reader(file, delimiter=" "))
file.close()

img_dir = "map-proj-v3-classdirs"

import os, shutil

for i in range(0, len(data)):
	src = img_dir + "/" + data[i][0]
	dst = img_dir + "/" + data[i][1] + "/" + data[i][0]
	try:
		shutil.move(src, dst)
	except:
		os.mkdir(img_dir + "/" + data[i][1])
	print(dst)
