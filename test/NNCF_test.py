import argparse
import sys

from dataloaders.datasets.supervisely import SUPSegmentation
from modeling.deeplab import DeepLab

if sys.platform == "win32":
    import distutils.command.build_ext
    import os
    from pathlib import Path

    VS_INSTALL_DIR = r"D:\\Program Files (x86)\\Microsoft Visual Studio"
    cl_paths = sorted(list(Path(VS_INSTALL_DIR).glob("**/Hostx86/x64/cl.exe")))
    if len(cl_paths) == 0:
        raise ValueError(
            "Cannot find Visual Studio. This notebook requires C++. If you installed "
            "a C++ compiler, please add the directory that contains cl.exe to "
            "`os.environ['PATH']`"
        )
    else:
        # If multiple versions of MSVC are installed, get the most recent version
        cl_path = cl_paths[-1]
        vs_dir = str(cl_path.parent)
        os.environ["PATH"] += f"{os.pathsep}{vs_dir}"
        # Code for finding the library dirs from
        # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
        d = distutils.core.Distribution()
        b = distutils.command.build_ext.build_ext(d)
        b.finalize_options()
        os.environ["LIB"] = os.pathsep.join(b.library_dirs)
        print(f"Added {vs_dir} to PATH")

import logging
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import List, Tuple

from onnx import load_model, save_model
from onnxmltools.utils import float16_converter
import torch
import nncf
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from nncf import NNCFConfig  # Important - should be imported directly after torch
from nncf.common.utils.logger import set_log_level

from torch.utils.data import DataLoader

set_log_level(logging.ERROR)  # Disables all NNCF info and warning messages
# from nncf.torch import create_compressed_model, register_default_init_args

# from openvino.runtime import Core
from torch.jit import TracerWarning

sys.path.append("../utils")
# from notebook_utils import download_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")
BASE_MODEL_NAME = "deeplabv3+"
IMAGE_SIZE = [288, 480]

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Paths where PyTorch, ONNX and OpenVINO IR models will be stored
fp32_checkpoint_filename = Path(BASE_MODEL_NAME + "_fp32").with_suffix(".pth")
fp32_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".onnx")
fp32_ir_path = fp32_onnx_path.with_suffix(".xml")
int8_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".onnx")
int8_ir_path = int8_onnx_path.with_suffix(".xml")

# fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/304_resnet50_fp32.pth"


# download_file(fp32_pth_url, directory=MODEL_DIR, filename=fp32_checkpoint_filename)


# def download_tiny_imagenet_200(
#         output_dir: Path,
#         url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
#         tarname: str = "tiny-imagenet-200.zip",):
#     archive_path = output_dir / tarname
#     download_file(url, directory=output_dir, filename=tarname)
#     zip_ref = zipfile.ZipFile(archive_path, "r")
#     zip_ref.extractall(path=output_dir)
#     zip_ref.close()
#     print(f"Successfully downloaded and extracted dataset to: {output_dir}")


# def create_validation_dir(dataset_dir: Path):
#     VALID_DIR = dataset_dir / "val"
#     val_img_dir = VALID_DIR / "images"
#
#     fp = open(VALID_DIR / "val_annotations.txt", "r")
#     data = fp.readlines()
#
#     val_img_dict = {}
#     for line in data:
#         words = line.split("\t")
#         val_img_dict[words[0]] = words[1]
#     fp.close()
#
#     for img, folder in val_img_dict.items():
#         newpath = val_img_dir / folder
#         if not newpath.exists():
#             os.makedirs(newpath)
#         if (val_img_dir / img).exists():
#             os.rename(val_img_dir / img, newpath / img)


# DATASET_DIR = OUTPUT_DIR / "tiny-imagenet-200"
# if not DATASET_DIR.exists():
#     download_tiny_imagenet_200(OUTPUT_DIR)
#     create_validation_dir(DATASET_DIR)

# def download_tiny_imagenet_200(
#         output_dir: Path,
#         url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
#         tarname: str = "tiny-imagenet-200.zip",
# ):
#     archive_path = output_dir / tarname
#     download_file(url, directory=output_dir, filename=tarname)
#     zip_ref = zipfile.ZipFile(archive_path, "r")
#     zip_ref.extractall(path=output_dir)
#     zip_ref.close()
#     print(f"Successfully downloaded and extracted dataset to: {output_dir}")


# def create_validation_dir(dataset_dir: Path):
#     VALID_DIR = dataset_dir / "val"
#     val_img_dir = VALID_DIR / "images"
#
#     fp = open(VALID_DIR / "val_annotations.txt", "r")
#     data = fp.readlines()
#
#     val_img_dict = {}
#     for line in data:
#         words = line.split("\t")
#         val_img_dict[words[0]] = words[1]
#     fp.close()
#
#     for img, folder in val_img_dict.items():
#         newpath = val_img_dir / folder
#         if not newpath.exists():
#             os.makedirs(newpath)
#         if (val_img_dir / img).exists():
#             os.rename(val_img_dir / img, newpath / img)


# DATASET_DIR = OUTPUT_DIR / "tiny-imagenet-200"
# if not DATASET_DIR.exists():
#     # download_tiny_imagenet_200(OUTPUT_DIR)
#     create_validation_dir(DATASET_DIR)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self, name: str, fmt: str = ":f"):
#         self.name = name
#         self.fmt = fmt
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val: float, n: int = 1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#     def __str__(self):
#         fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
#         return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     """Displays the progress of validation process"""
#
#     def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix
#
#     def display(self, batch: int):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print("\t".join(entries))
#
#     def _get_batch_fmtstr(self, num_batches: int):
#         num_digits = len(str(num_batches // 1))
#         fmt = "{:" + str(num_digits) + "d}"
#         return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


# def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module):
#     """Compute the metrics using data from val_loader for the model"""
#     batch_time = AverageMeter("Time", ":3.3f")
#     top1 = AverageMeter("Acc@1", ":2.2f")
#     top5 = AverageMeter("Acc@5", ":2.2f")
#     progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")
#
#     # switch to evaluate mode
#     model.eval()
#     model.to(device)
#
#     with torch.no_grad():
#         end = time.time()
#         for i, (images, target) in enumerate(val_loader):
#             images = images.to(device)
#             target = target.to(device)
#
#             # compute output
#             output = model(images)
#
#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             print_frequency = 10
#             if i % print_frequency == 0:
#                 progress.display(i)
#
#         print(
#             " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
#         )
#     return top1.avg


# def create_model(model_path: Path):
#     """Creates the ResNet-50 model and loads the pretrained weights"""
#     model = models.resnet50()
#     # update the last FC layer for Tiny ImageNet number of classes
#     NUM_CLASSES = 200
#     model.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)
#     model.to(device)
#     if model_path.exists():
#         checkpoint = torch.load(str(model_path), map_location="cpu")
#         model.load_state_dict(checkpoint["state_dict"], strict=True)
#     else:
#         raise RuntimeError("There is no checkpoint to load")
#     return model


# model = create_model(MODEL_DIR / fp32_checkpoint_filename)

DATASET_DIR = 'E:\\Project\\python_project\\deeplab-xception\\test\\tiny-imagenet-200'


def create_dataloaders(batch_size: int = 128):
    """Creates train dataloader that is used for quantization initialization and validation dataloader for computing the model accruacy"""
    train_dir = DATASET_DIR + "/train"
    val_dir = DATASET_DIR + "/val"  # + "/images"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = create_dataloaders()
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.base_size = 480
    # args.crop_size = [480, 288]
    #
    # sup_train = SUPSegmentation(args, split='train')  # 训练集的原图像
    # sup_val = SUPSegmentation(args, split='val')

    # train_loader = DataLoader(sup_train, batch_size=4, shuffle=True, num_workers=0)  # 只有主进程去加载batch数据
    # val_loader = DataLoader(sup_train, batch_size=1, shuffle=True, num_workers=0)  # 只有主进程去加载batch数据

    # acc1 = validate(val_loader, model)
    # print(f"Test accuracy of FP32 model: {acc1:.3f}")
    #
    # dummy_input = torch.randn(1, 3, *IMAGE_SIZE).to(device)
    # torch.onnx.export(model, dummy_input, fp32_onnx_path, opset_version=11)
    # print(f"FP32 ONNX model was exported to {fp32_onnx_path}.")

    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
    checkpoint = torch.load(
        'E:\\Project\\python_project\\deeplab-xception\\run\\supervisely\\deeplab-mobilenet\\model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    input = torch.rand(1, 3, 288, 480)
    output = model(input)
    print(output.size())
    torch.onnx.export(model, input, 'deeplab_mobilenet.onnx', verbose=True, opset_version=11)
    onnx_model = load_model('deeplab_mobilenet.onnx')
    trans_model = float16_converter.convert_float_to_float16(onnx_model)
    save_model(trans_model, "deeplab_mobilenet_fp16.onnx")


    # nncf_config_dict = {
    #     "input_info": {"sample_size": [1, 3, *IMAGE_SIZE]},
    #     "log_dir": str(OUTPUT_DIR),
    #     "compression": {
    #         "algorithm": "quantization",
    #         "initializer": {
    #             "range": {"num_init_samples": 15000},
    #             "batchnorm_adaptation": {"num_bn_adaptation_samples": 4000},
    #         },
    #     },
    # }

    # nncf_config = NNCFConfig.from_dict(nncf_config_dict)

    # nncf_config = register_default_init_args(nncf_config, train_loader)

    # compression_ctrl, model = create_compressed_model(model, nncf_config)

    # acc1 = validate(val_loader, model)
    # print(f"Accuracy of initialized INT8 model: {acc1:.3f}")

    # warnings.filterwarnings("ignore", category=TracerWarning)  # Ignore export warnings
    # warnings.filterwarnings("ignore", category=UserWarning)
    # compression_ctrl.export_model(int8_onnx_path)
    # print(f"INT8 ONNX model exported to {int8_onnx_path}.")

    # input_shape = [1, 3, *IMAGE_SIZE]
    # if not fp32_ir_path.exists():
    #     !mo --input_model "$fp32_onnx_path" --input_shape "$input_shape" --mean_values "[123.675, 116.28 , 103.53]" --scale_values "[58.395, 57.12 , 57.375]" --data_type FP16 --output_dir "$OUTPUT_DIR"
    #     assert fp32_ir_path.exists(), "The IR of FP32 model wasn't created"
