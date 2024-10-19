import h5py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


clean_img_dir = "../tmp/no_wm"
img_paths = os.listdir(clean_img_dir)

print(len(img_paths))

n_train = int(len(img_paths) * 0.8)


def is_black_img(img):
    if np.sum(img) == 0:
        return True
    return False


def is_white_img(img):
    if np.sum(img) // 255 // 3 == img.size:
        return True
    return False


with h5py.File("tmp/train.h5", "w") as h5f:
    train_num = 0
    bimg, wimg = False, False
    for file in tqdm(img_paths[:n_train]):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(clean_img_dir, file))
            img = np.transpose(img, (2, 0, 1))
            # if width or height < 256 -> padding
            if img.shape[1] < 256:
                # padding to 256
                pad_width = 256 - img.shape[1]
                _img = np.zeros((img.shape[0], 256, img.shape[2]))
                _img[:, :img.shape[1], :] = img
                img = _img
            if img.shape[2] < 256:
                # padding to 256
                pad_height = 256 - img.shape[2]
                _img = np.zeros((img.shape[0], img.shape[1], 256))
                _img[:, :, :img.shape[2]] = img
                img = _img
            # if width or height > 256 -> split into many images with overlap
            if img.shape[1] > 256 or img.shape[2] > 256:
                # split into many images with overlap
                # width = height = 256, hop = ...
                for i in range(0, img.shape[1] - 256 + 1, 192):
                    for j in range(0, img.shape[2] - 256 + 1, 192):
                        _img = img[:, i:i + 256, j:j + 256]

                        if is_black_img(_img):
                            if bimg:
                                print("skipping black")
                                continue
                            bimg = True
                        if is_white_img(_img):
                            if wimg:
                                print("skipping white")
                                continue
                            wimg = True
                        # print(_img.shape)
                        # plt.imshow(_img.transpose(1, 2, 0))
                        # plt.show()
                        h5f.create_dataset(str(train_num), data=_img)
                        train_num += 1
        # break