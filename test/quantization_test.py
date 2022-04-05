import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import make_data_loader
from dataloaders.datasets import supervisely
from dataloaders.datasets.supervisely import SUPSegmentation
from modeling.deeplab import DeepLab

model = DeepLab(num_classes=2,  # num_classes=8
                backbone='mobilenet',  # backbone=resnet
                output_stride=16,  # output_stride=16
                sync_bn=False,  # sync_bn= False
                freeze_bn=False)  # freeze_bn=False
checkpoint = torch.load('..\\run\\supervisely\\deeplab-mobilenet\\experiment_4\\checkpoint.pth.tar')
checkpoint_int8 = torch.load('mobilenet-qint8.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu1']], inplace=True)
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
    if i > 20:
        break
model_prepared_int8 = torch.quantization.convert(model_prepared)
torch.save(model_prepared_int8.state_dict(), "mobilenet-qint8.pth.tar")
