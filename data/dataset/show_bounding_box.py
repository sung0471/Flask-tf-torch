import os
import cv2
import numpy as np
from PIL import Image

dataset_dir = './box_dataset2'
folder = 'IsOccluded'
image_dir = os.path.join(dataset_dir, folder)
label_dir = os.path.join(dataset_dir, folder)
file_name = '0b2a53cb1c6b1b1c'

img_path = os.path.join(image_dir, file_name + '.jpg')
label_path = os.path.join(label_dir, file_name + '.txt')

img = Image.open(img_path)
w, h = img.size
img = np.array(img)
with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        coor = list(map(float, line.split()))
        if folder == 'cardboard':
            coor = coor[1:]
        xywh = False
        if not xywh:
            # c_x, c_y, w, h
            # (x1 + x2) / 2 = cx
            # (y1 + y2) / 2 = cy
            # x2 - x1 = w
            # y2 - y1 = h
            x_min = int((coor[0] - coor[2] / 2) * w)
            x_max = int((coor[0] + coor[2] / 2) * w)
            y_min = int((coor[1] - coor[3] / 2) * h)
            y_max = int((coor[1] + coor[3] / 2) * h)
        else:
            # x1, y1, x2, y2
            x_min = int(coor[0] * w)
            y_min = int(coor[1] * h)
            x_max = int(coor[2] * w)
            y_max = int(coor[3] * h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
Image.fromarray(img).show()
