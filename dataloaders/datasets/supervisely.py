from __future__ import print_function, division
import os

import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class SUPSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    # NUM_CLASSES = 21
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('supervisely'),  # 在mypath.py里面有设置，为数据集的路径 'G:\\LoveDA\\'
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir  # 数据集的根目录 'E:/CodeDownload/dataset/supervisely/'
        self._image_dir = os.path.join(self._base_dir, 'img')  # 'G:\\LoveDA\\JPEGImages'
        self._mask_dir = os.path.join(self._base_dir, 'mask')  # 'G:\\LoveDA\\SegmentationClass'

        if isinstance(split, str):  # split为‘train’训练集，'val'验证集  isinstance（）：判断两个类型是否相同
            self.split = [split]
        else:
            split.sort()  # 对列表进行原址排序
            self.split = split

        self.args = args

        # _ann_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')  # 存放着标签名称的文件夹路径
        # 'G:\\LoveDA\\ImageSets\\Segmentation'

        self.im_ids = []  # 存标签名的数组
        self.images = []  # 存图像地址的数组
        self.masks = []  # 存标签图片地址的数组

        for splt in self.split:  # 将图片信息读取到数组中
            with open(os.path.join(os.path.join(self._base_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()  # f.readlines()后面有加\n,f.read().splitlines()没有\n

            for ii, line in enumerate(lines):  # (ii=0,line='1366')  enumerate():返回 enumerate(枚举) 对象
                _image = os.path.join(self._image_dir, line)
                _mask = os.path.join(self._mask_dir, line)
                # assert检查条件，不符合就终止程序
                assert os.path.isfile(_image)  # os.path.isfile 用于判断某一对象(需提供绝对路径)是否为文件
                assert os.path.isfile(_mask)
                self.im_ids.append(line)  # 存入标签名  1366
                self.images.append(_image)  # 存入图像地址
                self.masks.append(_mask)  # 存入标签图片地址

        assert (len(self.images) == len(self.masks))  # 判断图像个数和标签数量是否一致

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))  # 1156

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)  # 读取图像 img:RGB形式  (1024, 1024)
        sample = {'image': _img, 'label': _target}  # {}字典数据类型，每一个键值对

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)  # 返回训练集的图像处理
            elif split == 'val':
                return self.transform_val(sample)  # 返回验证集的图像处理

    def _make_img_gt_point_pair(self, index):  # 仅仅是通过这个读就已经转成了0-7的标签图
        # Image.open()函数只是保持了图像被读取的状态，但是图像的真实数据并未被读取
        # Image.open()函数默认彩色图像读取通道的顺序为RGB
        # 如果不使用.convert(‘RGB’)进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
        _img = Image.open(self.images[index]).convert('RGB')  # 原图像    RGB: 3x8位像素，真彩色
        _mask = Image.open(self.masks[index])  # 标注图像

        # _img.show()
        # _target.show()

        return _img, _mask

    def transform_tr(self, sample):  # 返回训练集的图像处理：翻转-切割-平滑-标准化-tensor形式保存  大小控制在[-1.5, 0.5] 区间为2
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),  # 以给定的概率随机水平翻转给定的PIL图像
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),  # 将原图片进行切割随机范围内裁剪
            tr.RandomGaussianBlur(),  # 用高斯滤波器（GaussianFilter）对图像进行平滑处理
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 图片归一化
            tr.ToTensor()])  # tensor中是以 (c, h, w) 的格式来存储图片的
        result = composed_transforms(sample)
        # result['label'] = result['label'][:, :, 0].unsqueeze(dim=0)//255
        result['label'] = result['label'][:, :, 0] / 255
        result['label'] = torch.floor(result['label'])

        # return composed_transforms(sample)
        return result

    def transform_val(self, sample):  # 验证集的图片处理：切割-标准化-tensor形式保存 大小控制在[0，  7] 区间为8 因为图片的类别数为8

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        result = composed_transforms(sample)
        result['label'] = result['label'][:, :, 0] / 255
        result['label'] = torch.floor(result['label'])

        # return composed_transforms(sample)
        return result

    def __str__(self):
        return 'Supervisely_Person_Dataset(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 480
    args.crop_size = [480, 288]

    sup_train = SUPSegmentation(args, split='train')  # 训练集的原图像

    dataloader = DataLoader(sup_train, batch_size=4, shuffle=True, num_workers=0)  # 只有主进程去加载batch数据

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):  # 字典sample  jj:0~5 5个
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            plt.show()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='supervisely')  # mask上色
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # 将归一化的图像恢复原始数值，即0-255
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)  # 无符号八位整数，数字范围[0,255]
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)  # 显示原图像
            plt.subplot(212)
            plt.imshow(segmap)  # 显示彩色
            # print(segmap)

        if ii == 1:
            break

    plt.show(block=True)
