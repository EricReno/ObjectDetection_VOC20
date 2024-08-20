import os
import cv2
import numpy as np

import random

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.mp4', fourcc, 1, (960, 540))

data_list = []
for line in os.listdir('Non_Fire'):
    data_list.append(os.path.join(os.getcwd(), 'Non_Fire', line))

for line in os.listdir('Fire'):
    data_list.append(os.path.join(os.getcwd(), 'Fire', line))


random.shuffle(data_list)

for i in range(len(data_list)):
    image = cv2.imread(data_list[i])

    image = cv2.resize(image, (960, 540))

    video.write(image)

video.release(              )
cv2.destroyAllWindows()