import torch
import cv2
import numpy as np

from lenet5 import LeNet5

model = LeNet5()
model.load_state_dict(torch.load('lenet5.pth'))

image_path = 'data/3.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# add padding to 32x32
width, height = image.shape
default_size = 28
if width > height:
    padding = (width - height) // 2
    image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=255)
image = cv2.resize(image, (default_size, default_size), interpolation=cv2.INTER_AREA)

# negative of image if background is white
if np.mean(image) > 127:
    image = 255 - image

# image[image > 90] = 255

model.eval()
tensor_image = torch.from_numpy(image.astype(np.float32)).view(1, 1, default_size, default_size)
logits = model(tensor_image)
print(logits.argmax(dim=1).item())
