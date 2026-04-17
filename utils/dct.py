import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def mydct(img):

    h, w = img.shape[:2]
    # 读取图像
    # b, g, r = cv2.split(img)
    # img = cv2.merge((r, g, b))

    img1 = img[:, :, 0]
    img2 = img[:, :, 1]
    img3 = img[:, :, 2]

    # 数据类型转换 转换为浮点型
    # print('0\n', img)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img3 = img3.astype(np.float32)

    # 进行离散余弦变换
    img_dct1 = cv2.dct(img1)
    img_dct2 = cv2.dct(img2)
    img_dct3 = cv2.dct(img3)
    # print('1\n', img_dct)

    keep_h = max(1, int(h * 0.8))
    keep_w = max(1, int(w * 0.8))

    img_dct1[keep_h:, keep_w:] = 0
    img_dct2[keep_h:, keep_w:] = 0
    img_dct3[keep_h:, keep_w:] = 0

    img_idct1 = cv2.idct(img_dct1)
    img_idct2 = cv2.idct(img_dct2)
    img_idct3 = cv2.idct(img_dct3)

    nowimg = np.stack([img_idct1, img_idct2, img_idct3])
    nowimg = np.transpose(nowimg, (1, 2, 0))
    nowimg = nowimg.astype(np.uint8)

    return nowimg


