from lenet5 import load_data

from torchvision import transforms

import numpy as np
import cv2
import tqdm


def augment_data(x, y):
    kernel = np.ones((2, 2), np.uint8)
    print('Augmenting data...')
    augment_1 = [cv2.erode(i.reshape(28, 28).astype(np.uint8), kernel, iterations=1) for i in x]
    augment_1 = np.array(augment_1)
    augment_1 = augment_1.reshape(-1, 28 * 28)
    augment_2 = []
    for i in augment_1:
        new_im = i.copy()
        new_im[new_im > 20] = 255
        augment_2.append(new_im)
    augment_2 = np.array(augment_2)

    # save augmented data to csv
    def save_data(path, x, y):
        with open(path, 'w') as f:
            f.write('label,' + ','.join(['pixel' + str(i) for i in range(28 * 28)]) + '\n')
            for i in range(len(x)):
                f.write(f'{y[i]},{",".join([str(j) for j in x[i]])}\n')

    save_data('data/augment_1.csv', augment_1, y)
    save_data('data/augment_2.csv', augment_2, y)

    to_pil = transforms.ToPILImage()
    transform = transforms.RandomAffine(degrees=20)

    print('Transforming data...')

    for _ in range(2):
        augment_3 = []
        for i in tqdm.tqdm(x):
            new_im = i.reshape(28, 28).astype(np.uint8)
            new_im = transform(to_pil(new_im))
            augment_3.append(new_im)
        augment_3 = [np.array(i) for i in augment_3]
        augment_3 = np.array(augment_3)
        augment_3 = augment_3.reshape(-1, 28 * 28)
        save_data(f'data/augment_3_{_}.csv', augment_3, y)


if __name__ == '__main__':
    data_path = 'data'
    x, y = load_data(data_path, is_train=True)
    augment_data(x, y)
