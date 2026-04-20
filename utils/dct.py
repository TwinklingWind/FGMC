import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mydct(img):
    # 读取图像
    # b, g, r = cv2.split(img)
    # img = cv2.merge((r, g, b))

    keep_fraction = 0.5

    h, w = img.shape[:2]
    keep_h = max(1, int(h * keep_fraction))
    keep_w = max(1, int(w * keep_fraction))

    img1 = img[:, :, 0]
    img2 = img[:, :, 1]
    img3 = img[:, :, 2]

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img3 = img3.astype(np.float32)


    img_dct1 = cv2.dct(img1)
    img_dct2 = cv2.dct(img2)
    img_dct3 = cv2.dct(img3)
    # print('1\n', img_dct)


    dct_mask1 = np.zeros_like(img_dct1)
    dct_mask1[:keep_h, :keep_w] = img_dct1[:keep_h, :keep_w]
    dct_mask1[:keep_h, keep_w:] = img_dct1[:keep_h, keep_w:] * 0.8
    dct_mask1[keep_h:, :] = img_dct1[keep_h:, :] * 0.8


    dct_mask2 = np.zeros_like(img_dct2)
    dct_mask2[:keep_h, :keep_w] = img_dct2[:keep_h, :keep_w]
    dct_mask2[:keep_h, keep_w:] = img_dct2[:keep_h, keep_w:] * 0.8
    dct_mask2[keep_h:, :] = img_dct2[keep_h:, :] * 0.8


    dct_mask3 = np.zeros_like(img_dct3)
    dct_mask3[:keep_h, :keep_w] = img_dct3[:keep_h, :keep_w]
    dct_mask3[:keep_h, keep_w:] = img_dct3[:keep_h, keep_w:] * 0.8
    dct_mask3[keep_h:, :] = img_dct3[keep_h:, :] * 0.8

    img_idct1 = cv2.idct(dct_mask1)
    img_idct2 = cv2.idct(dct_mask2)
    img_idct3 = cv2.idct(dct_mask3)


    nowimg = np.stack([img_idct1, img_idct2, img_idct3])
    nowimg = np.transpose(nowimg, (1, 2, 0))
    nowimg = nowimg.astype(np.uint8)

    return nowimg
