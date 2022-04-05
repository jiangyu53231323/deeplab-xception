import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import make_data_loader
from dataloaders.datasets import supervisely
from dataloaders.datasets.supervisely import SUPSegmentation
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

model = DeepLab(num_classes=2,  # num_classes=8
                backbone='mobilenet',  # backbone=resnet
                output_stride=16,  # output_stride=16
                sync_bn=False,  # sync_bn= False
                freeze_bn=False)  # freeze_bn=False
model.eval()
# model_quantize = DeepLab(num_classes=2,  # num_classes=8
#                 backbone='mobilenet',  # backbone=resnet
#                 output_stride=16,  # output_stride=16
#                 sync_bn=False,  # sync_bn= False
#                 freeze_bn=False)  # freeze_bn=False
checkpoint = torch.load('..\\run\\supervisely\\deeplab-mobilenet\\experiment_4\\checkpoint.pth.tar')
# checkpoint_int8 = torch.load('mobilenet-qint8.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
# quantize_model(model_quantize, 'fbgemm')
# model_quantize.load_state_dict(checkpoint_int8)

torch.quantization.fuse_modules(model, [['Conv2d', 'BatchNorm2d', 'ReLU6']], inplace=True)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# model.fuse_model()
model_prepared = torch.quantization.prepare(model)
print(model_prepared)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_size = 480
args.crop_size = [480, 288]
kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}
# val_set = supervisely.SUPSegmentation(args, split='val')  # 读取val里面的图片信息
# val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
val_set = SUPSegmentation(args, split='val')  # 训练集的原图像
val_loader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=0)  # 只有主进程去加载batch数据

# model.eval()  # 测试状态
tbar = tqdm(val_loader, desc='\r')
test_loss = 0.0
for i, sample in enumerate(tbar):
    image, target = sample['image'], sample['label']
    # image, target = image.cuda(), target.cuda()
    with torch.no_grad():
        output = model_prepared(image)
    if i > 100:
        break
model_prepared_int8 = torch.quantization.convert(model_prepared)
torch.save(model_prepared_int8.state_dict(), "mobilenet-qint8.pth.tar")
