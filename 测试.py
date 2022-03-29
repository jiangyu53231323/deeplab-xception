import numpy as np
import cv2
import os
from PIL import Image

# BGR
color_segmentation = np.array([[255,255,255],[255,0,0],[255,255,0],
                     [0,0,255],[159,129,183],[0,255,0],
                     [255,195,128]], dtype=np.uint8)

def decode_segmap(label_mask, n_classes=7):  #(1024, 1024)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        # aaa = color_segmentation[ll, 1]
        # position = label_mask == ll
        b[label_mask == ll] = color_segmentation[ll, 0]  # 返回的相当于一个mask，只在true的位置上填充后面的值
        g[label_mask == ll] = color_segmentation[ll, 1]
        r[label_mask == ll] = color_segmentation[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    # print(rgb[240][240])

    # rgb[:, :, 0] = b
    # rgb[:, :, 1] = g
    # rgb[:, :, 2] = r

    rgb[:, :, 0] = b / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = r /255.0

    rgb = rgb.astype(np.uint8)  ##重要！ 要不然opencv显示有问题

    # print(rgb[240][240])

    return rgb


root_dir = "G:/LoveDA/SegmentationClass/"
list_gray_img = os.listdir(root_dir)
for img_name in list_gray_img:
    path_gray = root_dir + img_name
    laber_mask = cv2.imread(path_gray, 0)  #灰度 单通道读取
    # print(laber_mask.shape)
    color_img = decode_segmap(laber_mask)
    # print(color_img.shape)

    # img1 = Image.fromarray(color_img.astype('uint8')).convert('RGB')
    # img1.show()

