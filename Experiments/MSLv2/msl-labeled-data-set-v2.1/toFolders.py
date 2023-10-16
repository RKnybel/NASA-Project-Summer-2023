# This script splits the MSL dataset into train, validation and test sets according to the included .txt files. It also organizes them into class folders.

import csv
src_folder = "images"
dst_folder = "MSLv2-test"

file = open("test-set-v2.1.txt", "r")
data = list(csv.reader(file, delimiter=" "))
file.close()

print(data)

import os, shutil

#os.mkdir(dst_folder)
for i in range(0, len(data)):
	if not os.path.exists(dst_folder + "/" + data[i][1]):
		os.mkdir(dst_folder + "/" + data[i][1])
	shutil.copy(src_folder + "/" + data[i][0], dst_folder + "/" + data[i][1] +"/" + data[i][0])
