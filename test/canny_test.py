import os
import cv2

img = cv2.imread('img_1.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_color = img
blur = cv2.GaussianBlur(img_gray, (3, 3), 0)  # 用高斯滤波处理原图像降噪
canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值
cv2.namedWindow("canny",0);#可调大小
cv2.namedWindow("1",0);#可调大小
cv2.imshow('1', img)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
#
# def CannyThreshold(lowThreshold):
#     detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
#     detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
#     dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo', dst)
#
#
# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3
#
# img = cv2.imread('img_1.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.namedWindow('canny demo')
#
# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
#
# CannyThreshold(0)  # initialization
#
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()