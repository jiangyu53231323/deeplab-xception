import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):  # 当没有值的时候 mean std默认值
        self.mean = mean  # (0.485, 0.456, 0.406)
        self.std = std  # (0.229, 0.224, 0.225)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0  # 图片保存都是0~255的数值范围  将数值大小降到[0, 1]
        img -= self.mean  # [0, 0.5]
        img /= self.std  # [1, 2]

        return {'image': img,
                'label': mask}


class ToTensor(object):  # 图像从 np.array 转换为 tensor
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))  # 将图片的维度进行转换 在 tensor 中是以 (c, h, w) 的格式来存储图片的。
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):  # 随机裁剪
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size  # 513
        self.crop_size = crop_size  # 513
        self.fill = fill  # 0

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.7), int(self.base_size * 1.3))  # 返回 [257, 1026] 之间的任意整数  644
        w, h = img.size  # h:1024  w:1024 原图的尺寸大小
        if h > w:
            ow = short_size  # ow：644
            oh = int(1.0 * h * ow / w)  # oh:644
        else:
            oh = short_size  # oh 307
            ow = int(1.0 * w * oh / h)  # w = h so ow = oh = 644
        # resize函数来改变图像的大小
        # Image.NEAREST ：低质量  最邻近插值
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        img = img.resize((ow, oh), Image.BILINEAR)  # 利用双线性插值公式的方法来进行图像的缩放
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop 填充
        if ow < self.crop_size[0] or oh < self.crop_size[1]:  # 随机裁剪大小 小于 目标裁剪大小时
            padh = self.crop_size[1] - oh if oh < self.crop_size[1] else 0  # 513-307=206
            padw = self.crop_size[0] - ow if ow < self.crop_size[0] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # 填充  fill=0默认值为0，表示颜色为黑色
            # (513, 513) border的len为4，left, top, right, bottom右边和底部进行填充：0 黑色的像素值 然后图片大小变成（513，513）
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size  # 随机裁剪的大小 （644，644）
        x1 = random.randint(0, w - self.crop_size[0])  # x1:44  (0, 644-513 = 131)之间的任意整数
        y1 = random.randint(0, h - self.crop_size[1])  # y1:34  (0,131)之间的任意整数
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))  # 切割
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size  # h:1024  w:1024 原图的尺寸大小
        if w > h:
            oh = self.crop_size[0]
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size[1]  # 513
            oh = int(1.0 * h * ow / w)  # oh = ow = 513
        img = img.resize((ow, oh), Image.BILINEAR)  # 利用双线性插值公式的方法来进行图像的缩放(513,513)
        mask = mask.resize((ow, oh), Image.NEAREST)  # 利用最近邻插值公式的方法来进行图像的缩放
        # center crop
        w, h = img.size  # (513,513)
        x1 = int(round((w - self.crop_size[0]) / 2.))  # x1=0
        y1 = int(round((h - self.crop_size[1]) / 2.))  # y1=0
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))  # (513, 513)  (0,0,513,513)
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
