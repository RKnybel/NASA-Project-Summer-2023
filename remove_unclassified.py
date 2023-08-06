# Removes unclassified images in the MSL dataset defined in unclassified.txt

import os
img_folder = 'calibrated'

uc_file = open("unclassified.txt", "r")

for img in uc_file:
	img=img.strip("\n")
	os.remove(img_folder + "/" + img)
	print(img + " removed")
