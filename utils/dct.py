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



if __name__ == '__main__':
    # -------------------------- CONFIGURE YOUR FOLDER PATH HERE --------------------------
    image_directory = "./"  # Replace with your folder path (e.g., "C:/photos" or "./images")
    # --------------------------------------------------------------------------------------

    # Check if the directory exists
    if not os.path.isdir(image_directory):
        print(f"Error: Directory '{image_directory}' does not exist!")
    else:
        # Get list of all files in the directory
        image_files = os.listdir(image_directory)

        # Supported image extensions (add more if needed)
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

        # Counter for loaded images
        image_count = 0

        print(f"Reading images from: {image_directory}\n")

        # Loop through all files
        for filename in image_files:
            # Check if the file is an image
            if filename.lower().endswith(supported_formats):
                # Full path to the image file
                image_path = os.path.join(image_directory, filename)

                # Read image with OpenCV
                # cv2.IMREAD_COLOR = load color image (default)
                # cv2.IMREAD_GRAYSCALE = load grayscale
                # cv2.IMREAD_UNCHANGED = load with alpha channel
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)

                # Check if image loaded successfully
                if img is None:
                    print(f"⚠️ Could not read: {filename} (corrupted or not an image)")
                    continue

                # -------------------------- YOUR IMAGE PROCESSING HERE --------------------------
                # Example: Show image dimensions
                height, width, channels = img.shape
                print(f"✅ Loaded: {filename} | Size: {width}x{height} | Channels: {channels}")

                # Example: Display the image (press any key to close window)
                cv2.imshow(f"Image: {filename}", img)
                cv2.waitKey(0)  # Wait for key press
                cv2.destroyAllWindows()  # Close window
                # ------------------------------------------------------------------------------

                image_count += 1

        print(f"\nDone! Successfully loaded {image_count} images.")
