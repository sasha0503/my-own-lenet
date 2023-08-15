import random

import cv2
import numpy as np

data_path = 'data/train.csv'

data = np.loadtxt(data_path, dtype=str, delimiter=',')

data = data[1:, :]
y, x = data[:, 0], data[:, 1:]

samples = random.sample(range(len(x)), 100)
for i in samples:
    image = x[i].reshape(28, 28).astype(np.uint8)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    print(y[i])
    cv2.imshow('image', image)
    cv2.waitKey(0)
