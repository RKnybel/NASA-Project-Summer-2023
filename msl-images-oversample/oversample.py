# Duplicates images to "balance" the dataset

import os
import shutil

# get classes and remove this script
classes = os.listdir("./")
classes.remove("oversample.py")

# holds number of samples per class
class_dict = {}

max_samples = 0

# find the largest class and update # of samples
for class_name in classes:
	n_samples = len(os.listdir("./" + class_name))
	if max_samples < n_samples:
		max_samples = n_samples
	class_dict.update({class_name: n_samples})

# copy images to oversample
for class_name in classes:
	n_samples = class_dict[class_name]
	samples = os.listdir("./" + class_name)
	n_copies = max_samples // n_samples
	
	file_num = 1
	for i in range(1, n_copies):
		for sample_file in samples:
			if os.path.isfile("./" + class_name + "/" + sample_file):
        			shutil.copy("./" + class_name + "/" + sample_file, "./" + class_name + "/" + sample_file.replace(".JPG", "") + "_" + str(file_num) + ".JPG")
        			print(sample_file + "_" + str(file_num))
        			file_num += 1
	
print(max_samples)

print(class_dict)
