# Resnet

## 说明

本文构建了ResNet模型，包括模型结构、训练和评估等流程，方便初学者简单快速地完成一次十分类深度学习实验。该仓库下还有mobilenet模型的相关文件。

本代码来源于 https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test5_resnet

本代码在此基础上写了大量注释，更加便于学习

论文：https://arxiv.org/abs/1512.03385

## 文件结构：

```
  ├── model.py: ResNet模型搭建
  ├── train.py: 训练脚本
  ├── predict.py: 图像预测脚本
```

## 使用方法：

首先需要下载CIFAR-10-batches-py数据集，链接为

https://www.cs.toronto.edu/~kriz/cifar.html

该数据集图像是32*32*3因此无法使用预训练权重resnet34-333f7ec4.pth，下面仍然给出预训练权重下载url方便后续需要

根据需要的resnet模型，下载预训练权重，然后指定划分后的数据集路径（train脚本28行），指定权重路径（68行），运行train.py脚本。

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



## 训练数据

本模型使用的是cifar-10数据集，如果要使用其他分类数据集，需要重写dataset模块。十分类数据集下载链接：

https://www.cs.toronto.edu/~kriz/cifar.html

下载后运行train.py进行模型训练，可以将epoch调至100

## 测试数据：

有了resnet34_cifar10.pth后，我们运行predict.py对test_cifar10里的数据进行预测，得到结果保存至results.csv中





