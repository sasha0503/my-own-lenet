import torch
import cv2
import numpy as np

from lenet5 import LeNet5, load_data


def predict(image):
    # add padding to 32x32
    width, height = image.shape
    default_size = 28
    if width > height:
        padding = (width - height) // 2
        image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=255)
    elif height > width:
        padding = (height - width) // 2
        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=255)
    image = cv2.resize(image, (default_size, default_size), interpolation=cv2.INTER_AREA)

    tensor_image = torch.from_numpy(image.astype(np.float32)).view(1, 1, default_size, default_size)
    logits = model(tensor_image)
    print(logits.argmax(dim=1).item())


if __name__ == '__main__':
    model = LeNet5()
    model.load_state_dict(torch.load('models/lenet5_2023-08-17_17-51-43.pth'))
    model.eval()

    data_path = 'data'
    x = load_data(data_path, is_train=False)
    x = torch.from_numpy(x.astype(np.float32)).view(-1, 1, 28, 28)
    res = model(x)
    res = res.argmax(dim=1).numpy()
    with open("submission.csv", "w") as f:
        f.write("ImageId,Label\n")
        for i in range(len(res)):
            f.write(f"{i + 1},{res[i]}\n")
