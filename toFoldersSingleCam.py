# This script splits the MSL dataset into train, validation and test sets according to the included .txt files. It also organizes them into class folders.

import csv

file = open("classes.txt", "r")
data = list(csv.reader(file, delimiter=" "))
file.close()
img_dir = "MR"

print(data)

import os, shutil

for i in range(0, len(data)):
	if not os.path.exists(img_dir + "/" + data[i][1]):
		os.mkdir(img_dir + "/" + data[i][1])
	try:
		shutil.copy(img_dir + data[i][0].replace("calibrated", ""), img_dir + "/" + data[i][1] + data[i][0].replace("calibrated", ""))
	except:
		print(img_dir + data[i][0].replace("calibrated", "") + " does not exist.")
