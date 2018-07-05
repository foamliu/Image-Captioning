# 图像中文描述

用一句话描述给定图像中的主要信息，中文语境下的图像理解问题。尝试自然语言处理与计算机视觉的结合。

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

使用 AI Challenger 2017 的图像中文描述数据集，包含30万张图片，150万句中文描述。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。


 ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/dataset.png)

要下载点这里：图像中文描述 [数据集](https://challenger.ai/datasets/caption)，放在 data 目录下。


## ImageNet 预训练模型

下载 [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing)， 放在 models 目录下。

## 用法

### 数据预处理
提取60,999个训练图像，并将它们分开（53,879个用于训练，7,120个用于验证）：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

如果想在培训期间进行可视化，请在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [pre-trained model](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/2_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/3_out.png) |
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/6_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/7_out.png) |
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/9_out.png) |![image](https://github.com/foamliu/Scene-Classification/raw/master/images/10_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/11_out.png)|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/12_out.png)  | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/13_out.png) |![image](https://github.com/foamliu/Scene-Classification/raw/master/images/14_out.png)| ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/15_out.png)|
|![image](https://github.com/foamliu/Scene-Classification/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/17_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Scene-Classification/raw/master/images/19_out.png) |


### 结果分析

```bash
$ python analyze.py
```

| |Test A|Test B|
|---|---|---|
|图片数|7040|7078|
|Top3准确度|0.90511|0.86337|