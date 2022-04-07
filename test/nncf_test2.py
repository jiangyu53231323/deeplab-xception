import torch
import nncf  # Important - should be imported directly after torch

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args

# Instantiate your uncompressed model
from torchvision.models.resnet import resnet50

from modeling.deeplab import DeepLab

# model = resnet50()
model = DeepLab(backbone='mobilenet', output_stride=16)
model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
checkpoint = torch.load('.\model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
input = torch.rand(1, 3, 288, 480)
output = model(input)
print(output.size())
torch.onnx.export(model, input, 'deeplab_mobilenet.onnx', verbose=True, opset_version=11)

# Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_int8.json")

# Provide data loaders for compression algorithm initialization, if necessary
# import torchvision.datasets as datasets
# representative_dataset = datasets.ImageFolder("/path")
# init_loader = torch.utils.data.DataLoader(representative_dataset)
# nncf_config = register_default_init_args(nncf_config, init_loader)

# Apply the specified compression algorithms to the model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual torch.nn.Module
# to fine-tune compression parameters along with the model weights

# ... the rest of the usual PyTorch-powered training pipeline

# Export to ONNX or .pth when done fine-tuning
compression_ctrl.export_model("compressed_model.onnx")
torch.save(compressed_model.state_dict(), "compressed_model.pth")