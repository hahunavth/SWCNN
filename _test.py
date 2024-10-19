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




with h5py.File("tmp/val.h5", "w") as h5f:
    val_num = 0
    for file in tqdm(img_paths[n_train:]):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(clean_img_dir, file))
            img = np.transpose(img, (2, 0, 1))
            
            h5f.create_dataset(str(val_num), data=img)
            val_num += 1