from modeling.backbone import resnet, xception, drn, mobilenet, ghostnet, mobilenet_v3


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'ghostnet':
        return ghostnet.my_ghostnet(output_stride)
    elif backbone == 'mobilenetv3':
        return mobilenet_v3.MobileNetV3_Small()
    else:
        raise NotImplementedError
