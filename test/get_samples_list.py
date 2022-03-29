import glob
import os
import random

# list = os.listdir("C:\Program Files (x86)\Intel\openvino_2021.4.752\opencv\lib")
# for i in list:
#     if i.endswith("453.lib"):
#         print(i)
# print(list)

list_all = glob.glob('E:/CodeDownload/dataset/supervisely/img/*')
train_num = int(len(list_all) * 0.7)
train_list = random.sample(list_all, train_num)
test_list = list(set(list_all).difference(set(train_list)))
print(len(list_all))

with open("train.txt", "w") as f:
    for i in range(len(train_list)):
        f.write(str(train_list[i].split('\\')[-1]) + '\n')

with open("val.txt", "w") as f:
    for i in range(len(test_list)):
        f.write(str(test_list[i].split('\\')[-1]) + '\n')
