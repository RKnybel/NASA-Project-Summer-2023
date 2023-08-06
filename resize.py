# resizes all images in a folder to a specified size

import PIL
import os
from PIL import Image

images_folder = 'calibrated'
x = 256
y = 256

for file in os.listdir(images_folder):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((x,y))
    img.save(f_img)
