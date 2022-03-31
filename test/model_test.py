import os
import cv2
import numpy as np
import torch

from modeling.deeplab import DeepLab

if __name__ == "__main__":
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    checkpoint = torch.load(
        'E:\\Project\\python_project\\deeplab-xception\\run\\supervisely\\deeplab-mobilenet\\model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    file_mp4 = './TEST_04.mp4'
    video = cv2.VideoCapture(file_mp4)
    success, frame = video.read()
    frame_num = 1
    while (success):
        frame = cv2.resize(frame, (480, 288))
        cv2.imwrite('./TEST_04/frame' + str(frame_num) + '.png', frame)
        img = np.array(frame).astype(np.float32)
        img /= 255.0  # 图片保存都是0~255的数值范围  将数值大小降到[0, 1]
        img -= IMG_MEAN  # [0, 0.5]
        img /= IMG_STD  # [1, 2]
        img = img[:, :, ::-1].copy().transpose((2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(dim=0)
        output = model(img)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1).squeeze(axis=0)
        pred = np.expand_dims(pred, axis=2)
        img_seg = np.array(frame).astype(np.float32) * pred
        cv2.imwrite('./TEST_04/seg' + str(frame_num) + '.png', img_seg)
        success, frame = video.read()
        frame_num = frame_num + 1
