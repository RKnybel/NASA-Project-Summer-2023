# This script separates MSL images by camera, into three folders: MH, ML and MR.

import os, shutil

#Make folders
paths = ["MH", "ML", "MR"]
image_folder = "calibrated"
for path in paths:
	if not os.path.exists(path):
		os.makedirs(path)

# copy files to respective folders
image_count = 0
with os.scandir(image_folder) as images:
	for image in images:
		for path in paths:
			if image.name.find(path) != -1:
				src = image_folder + "/" + image.name
				dst = path + "/" + image.name
				print(src + " " + path)
				shutil.copy2(src, dst)
