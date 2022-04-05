import os
import cv2
import numpy as np
import torch

from modeling.my_deeplab import DeepLab


def quantize_model(model, backend):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()

    _dummy_input_data = torch.rand(1, 3, 288, 480)

    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_histogram_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_histogram_observer,
            weight=torch.quantization.default_weight_observer)

    # model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

if __name__ == "__main__":
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    # model = DeepLab(backbone='mobilenet', output_stride=16)
    model_quantize = DeepLab(backbone='mobilenet', output_stride=16)
    model_quantize.eval()
    # model.eval()
    # checkpoint = torch.load(
    #     'E:\\Project\\python_project\\deeplab-xception\\run\\supervisely\\deeplab-mobilenet\\model_best.pth.tar')
    checkpoint_int8 = torch.load('mobilenet-qint8.pth.tar')
    quantize_model(model_quantize, 'fbgemm')
    model_quantize.load_state_dict(checkpoint_int8)
    # model.load_state_dict(checkpoint['state_dict'])
    file_mp4 = './TEST_04.mp4'
    video = cv2.VideoCapture(file_mp4)
    success, frame = video.read()
    frame_num = 1
    while (success):
        frame = cv2.resize(frame, (480, 288))
        cv2.imwrite('./TEST_04_quantize/frame' + str(frame_num) + '.png', frame)
        img = np.array(frame).astype(np.float32)
        img /= 255.0  # 图片保存都是0~255的数值范围  将数值大小降到[0, 1]
        img -= IMG_MEAN  # [0, 0.5]
        img /= IMG_STD  # [1, 2]
        img = img[:, :, ::-1].copy().transpose((2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(dim=0)
        output = model_quantize(img)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1).squeeze(axis=0)
        pred = np.expand_dims(pred, axis=2)
        img_seg = np.array(frame).astype(np.float32) * pred
        cv2.imwrite('./TEST_04_quantize/seg' + str(frame_num) + '.png', img_seg)
        success, frame = video.read()
        frame_num = frame_num + 1
