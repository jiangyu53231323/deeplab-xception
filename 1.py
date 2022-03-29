# encoding:utf-8
import torch
import torch.nn as nn

# num_features - num_features from an expected input of size:batch_size*num_features*height*width
# eps:default:1e-5 (公式中为数值稳定性加到分母上的值)
# momentum:动量参数，用于running_mean and running_var计算的值，default：0.1
m = nn.BatchNorm2d(2, affine=True)  # affine参数设为True表示weight和bias将被使用
input = torch.randn(1, 2, 3, 4)
output = m(input)

print(input)
print(m.weight)
print(m.bias)
print(output)
print(output.size())


print("输入的第一个维度:")
print(input[0][0]) #这个数据是第一个3*4的二维数据
#求第一个维度的均值和方差
firstDimenMean=torch.Tensor.mean(input[0][0])
firstDimenVar=torch.Tensor.var(input[0][0],False)   #false表示贝塞尔校正不会被使用
print(m)
print('m.eps=',m.eps)
print(firstDimenMean)
print(firstDimenVar)


batchnormone=((input[0][0][0][0]-firstDimenMean)/(torch.pow(firstDimenVar,0.5)+m.eps))\
    *m.weight[0]+m.bias[0]
print(batchnormone)