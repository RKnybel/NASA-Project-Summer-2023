# This script splits the MSL dataset into train, validation and test sets according to the included .txt files. It also organizes them into class folders.

import csv

file = open("classes.txt", "r")
data = list(csv.reader(file, delimiter=" "))
file.close()

print(data)

import os, shutil

os.mkdir("images")
for i in range(0, len(data)):
	if not os.path.exists("images/" + data[i][1]):
		os.mkdir("images/" + data[i][1])
	shutil.copy(data[i][0], "images/" + data[i][1] + data[i][0].replace("calibrated", ""))
