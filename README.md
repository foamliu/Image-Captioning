# 图像中文描述

用一句话描述给定图像中的主要信息，中文语境下的图像理解问题。尝试自然语言处理与计算机视觉的结合。

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

使用 AI Challenger 2017 的图像中文描述数据集，包含30万张图片，150万句中文描述。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。


 ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/dataset.png)

下载点这里：[图像中文描述数据集](https://challenger.ai/datasets/caption)，放在 data 目录下。


## 网络结构

 ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/net.png)

## 用法

### 数据预处理
提取210,000 张训练图片和30,000 张验证图片：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

可视化训练过程，执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 演示
下载 [预训练模型](https://github.com/foamliu/Image-Captioning/releases/download/v1.0/model.07-1.5001.hdf5) 放在 models 目录，然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/2_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/3_image.png) |
| ﻿一群 穿着 球衣 的 人 在 运动场 上 打 排球 | 一个 右手 拿 着 话筒 的 男人 在 屋子里 唱歌 | 一个 双手 拿 着 东西 的 孩子 站 在 室外 的 台阶 上 | 一个 穿着 球衣 的 男人 在 球场上 踢足球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/6_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/7_image.png) |
| 擂台 上 有 两个 穿着 黑色 裤子 的 男人 在 打拳击 | 一个 戴着 帽子 的 男人 在 草地 上 打 高尔夫球 | 一个 穿着 红色 上衣 的 女人 蹲 在 花丛 中 | 一个 穿着 短袖 的 男人 抱 着 一个 穿着 裙子 的 女人 站 在 房间 里 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/9_image.png) |![image](https://github.com/foamliu/Image-Captioning/raw/master/images/10_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/11_image.png)|
| 室内 有 三个 穿着 各异 的 孩子 在 做 手工 | 大 棚里 有 两个 穿着 深色 裤子 的 女人 在 摘 辣椒 | 一个 穿着 黑色 上衣 的 男人 和 一个 穿着 白色 上衣 的 男人 在 篮球场 上 交谈 | 一个 背着 双肩包 的 男人 和 一个 穿着 裙子 的 女人 走 在 山上 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/12_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/13_image.png) |![image](https://github.com/foamliu/Image-Captioning/raw/master/images/14_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/15_image.png)|
| 一个 戴着 帽子 的 女人 和 一个 戴着 帽子 的 男人 走 在 草地 上| 一个 双手 拿 着 包 的 女人 站 在 平坦 的 道路 上 | 一个 穿着 黑色 裤子 的 男人 和 一个 穿着 裙子 的 女人 站 在 舞台 上 | 一个 穿着 古装 的 男人 牵着 一个 穿着 裙子 的 女孩 走 在 道路 上 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/16_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/17_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/18_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/19_image.png) |
| 一个 右手 拿 着 手机 的 女人 站 在 房屋 前 | 一个 穿着 西装 的 男人 和 一个 穿着 球衣 的 男人 走 在 运动场 上 | 房间 里 有 一个 穿着 短裤 的 女人 坐在 沙发 上 | 一个 穿着 红色 上衣 的 男人 在 室内 的 垫子 上 做 俯卧撑 |

### 光束搜索 (Beam Search)
下载 [预训练模型](https://github.com/foamliu/Image-Captioning/releases/download/v1.0/model.07-1.5001.hdf5) 放在 models 目录，然后执行:

```bash
$ python beam_search.py
```

1 | 2 |
|---|---|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/0_bs_image.png) | Normal Max search: 一个 双手 叉腰 的 女人 站 在 广告牌 前 的 地上<br>Beam Search, k=3: 广告牌 前站 着 一个 双手 放在 身前 的 女人<br>Beam Search, k=5: 发布会 的 幕布 前站 着 一个 双手 叉腰 的 女人<br>Beam Search, k=7: 明亮 的 舞台 上 站 着 一个 左手 叉腰 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/1_bs_image.png) | Normal Max search: 一个 穿着 短袖 上衣 的 男人 在 室内 的 桌子 旁 操作电脑<br>Beam Search, k=3: 房间 里 有 一个 右手 拿 着 笔 的 女人 在 写字<br>Beam Search, k=5: 屋子里 有 一个 右手 拿 着 笔 的 男人 在 写字<br>Beam Search, k=7: 屋子里 有 一个 右手 拿 着 笔 的 男人 在 写字 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/2_bs_image.png) | Normal Max search: 冰场 上 一个 人 的 前面 有 三个 戴着 头盔 的 人 在 打 冰球<br>Beam Search, k=3: 宽敞 的 滑冰场 上 有 一群 戴着 头盔 的 人 在 打 冰球<br>Beam Search, k=5: 宽敞 的 冰面 上 有 一群 戴着 头盔 的 人 在 打 冰球<br>Beam Search, k=7: 光滑 的 滑冰场 上 有 一群 戴着 头盔 的 人 在 打 冰球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/3_bs_image.png) | Normal Max search: 一个 穿着 比基尼 的 女人 站 在 游泳池 边 的 道路 上<br>Beam Search, k=3: 一个 穿着 比基尼 的 女人 站 在 泳池 边 的 道路 上<br>Beam Search, k=5: 一个 穿着 比基尼 的 女人 站 在 室外 的 泳池 旁<br>Beam Search, k=7: 清澈 的 泳池 边站 着 一个 穿着 比基尼 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/4_bs_image.png) | Normal Max search: 一个 穿着 裙子 的 女人 站 在 人群 熙攘 的 大厅 里<br>Beam Search, k=3: 室外 的 红毯 上 站 着 一个 右手 拿 着 包 的 女人<br>Beam Search, k=5: 铺 着 红毯 的 道路 上 站 着 一个 右手 拿 着 包 的 女人<br>Beam Search, k=7: 铺 着 红毯 的 道路 上 站 着 一个 右手 拿 着 包 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/5_bs_image.png) | Normal Max search: 两个 穿着 球衣 的 男人 在 球场上 踢足球<br>Beam Search, k=3: 足球场 上 有 两个 穿着 不同 球衣 的 男人 在 抢球<br>Beam Search, k=5: 碧绿 的 球场上 有 两位 穿着 球服 的 男士 在 抢 足球<br>Beam Search, k=7: 碧绿 的 球场上 有 两位 穿着 球服 的 男士 在 抢 足球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/6_bs_image.png) | Normal Max search: 两个 穿着 球衣 的 男人 在 球场上 打篮球<br>Beam Search, k=3: 球场上 有 两个 穿着 运动服 的 男人 在 打篮球<br>Beam Search, k=5: 球场上 有 两个 穿着 运动服 的 男人 在 打篮球<br>Beam Search, k=7: 平坦 的 球场上 有 两位 穿着 球服 的 男士 在 打篮球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/7_bs_image.png) | Normal Max search: 一个 戴着 墨镜 的 女人 站 在 室内 的 展板 前<br>Beam Search, k=3: 明亮 的 房间 里 站 着 一个 右手 拿 着 话筒 的 女人<br>Beam Search, k=5: 明亮 的 房间 里 站 着 一个 右手 拿 着 话筒 的 女人<br>Beam Search, k=7: 明亮 的 房间 里 站 着 一个 右手 拿 着 话筒 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/8_bs_image.png) | Normal Max search: 一个 双手 拿 着 锄头 的 男人 站 在 田地 里<br>Beam Search, k=3: 大 棚里 有 一个 双手 拿 着 工具 的 男人 在 干活<br>Beam Search, k=5: 大 棚里 有 一个 双手 拿 着 工具 的 男人 在 干活<br>Beam Search, k=7: 大 棚里 有 一个 双手 拿 着 工具 的 男人 在 干活 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/9_bs_image.png) | Normal Max search: 一个 戴着 帽子 的 男人 抱 着 一个 孩子 走 在 道路 上<br>Beam Search, k=3: 一个 戴着 帽子 的 男人 抱 着 一个 孩子 走 在 道路 上<br>Beam Search, k=5: 一个 戴着 帽子 的 男人 抱 着 一个 孩子 走 在 道路 上<br>Beam Search, k=7: 一个 戴着 帽子 的 男人 抱 着 一个 孩子 走 在 道路 上 |

### 模型评估

在 30000 张验证集图片上测得 BLEU-4 并求均值，得到：0.46908。

```bash
$ python bleu_scores.py
```
