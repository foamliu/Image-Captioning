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

 <img src="https://raw.githubusercontent.com/foamliu/Image-Captioning/master/model.py?sanitize=true">

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

### 性能

在 30000 张测试集 (test-a) 图片上测得 BLEU-4 并求均值，得到：0.64684。

```bash
$ python bleu_scores.py
```

### 演示
下载 [预训练模型](https://github.com/foamliu/Image-Captioning/releases/download/v1.0/model.04-1.3820.hdf5) 放在 models 目录，然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/2_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/3_image.png) |
| ﻿一个 穿着 裙子 的 女人 在 滑冰场 上 表演 花样滑冰 | 房间 里 一个 人 的 旁边 有 一个 穿着 短袖 的 男人 在 给 一个 穿着 红色 上衣 的 女人 递 东西 | 舞台 上 有 一个 右手 拿 着 话筒 的 女人 在 唱歌 | 房间 里 有 一个 穿着 白色 上衣 的 女人 在 给 一群 孩子 讲课 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/6_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/7_image.png) |
| 房间 里 有 一群 穿着 各异 的 孩子 在 玩耍 | 房间 里 有 一个 右手 拿 着 笔 的 女人 在 写字 | 一个 戴着 帽子 的 女人 走 在 人来人往 的 道路 上 | 一个 右手 拿 着 笔 的 女人 坐在 室内 的 桌子 旁 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/9_image.png) |![image](https://github.com/foamliu/Image-Captioning/raw/master/images/10_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/11_image.png)|
| 一个 穿着 短袖 的 男人 推 着 一辆 购物车 走 在 超市 里 | 一个 穿着 球衣 的 男人 在 球场上 踢足球 | 一个 右手 拿 着 话筒 的 女人 在 广告牌 前 讲话 | 两个 穿着 球衣 的 男人 在 球场上 踢足球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/12_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/13_image.png) |![image](https://github.com/foamliu/Image-Captioning/raw/master/images/14_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/15_image.png)|
| 房间 里 一个 人 的 旁边 有 一个 戴着 眼镜 的 男人 在 给 一个 穿着 深色 上衣 的 男人 递 东西| 一个 双手 拿 着 球杆 的 男人 在 滑冰场 上 滑冰 | 一个 戴着 眼镜 的 男人 和 一个 穿着 短袖 的 男人 在 室内 的 桌子 旁 交谈 | 一个 双手 拿 着 奖杯 的 女人 站 在 舞台 上 讲话 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/16_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/17_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/18_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/19_image.png) |
| 一个 赤裸 上身 的 男人 在 海边 的 沙滩 上 奔跑 | 两个 穿着 球衣 的 男人 在 球场上 打篮球 | 一群 穿着 泳衣 的 人 在 游泳池 里 游泳 | 一个 右手 拎 着 包 的 男人 走 在 道路 上 |

### 光束搜索 (Beam Search)
下载 [预训练模型](https://github.com/foamliu/Image-Captioning/releases/download/v1.0/model.04-1.3820.hdf5) 放在 models 目录，然后执行:

```bash
$ python beam_search.py
```

1 | 2 |
|---|---|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/0_bs_image.png) | Normal Max search: 一个 戴着 墨镜 的 女人 和 一个 穿着 牛仔裤 的 男人 走 在 道路 上<br>Beam Search, k=3: 一个 戴着 墨镜 的 女人 和 一个 右手 拎 着 包 的 男人 走 在 道路 上<br>Beam Search, k=5: 一个 右手 拎 着 包 的 女人 挽 着 一个 戴着 帽子 的 男人 走 在 道路 上<br>Beam Search, k=20: 平整 的 道路 上 走 着 两个 戴着 墨镜 的 人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/1_bs_image.png) | Normal Max search: 运动场 上 一个 人 的 前面 有 一个 穿着 西装 的 男人 在 给 一个 穿着 运动服 的 男人 按摩 腿<br>Beam Search, k=3: 运动场 上 有 两个 穿着 西装 的 男人 坐在 椅子 上 交谈<br>Beam Search, k=5: 运动场 上 有 两个 穿着 西装 的 男人 坐在 椅子 上 讲话<br>Beam Search, k=20: 碧绿 的 球场上 坐 着 一位 双手 搭 在 腿 上 的 男士 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/2_bs_image.png) | Normal Max search: 一个 穿着 运动服 的 男人 走 在 运动场 上<br>Beam Search, k=3: 足球场 上 走 着 一个 穿着 运动服 的 男人<br>Beam Search, k=5: 碧绿 的 球场上 走 着 一位 穿着 球服 的 男士<br>Beam Search, k=20: 平整 的 球场上 走 着 一位 穿着 球服 的 男士 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/3_bs_image.png) | Normal Max search: 一个 戴着 帽子 的 男人 和 一个 穿着 黑色 上衣 的 男人 在 室内 的 桌子 旁 交谈<br>Beam Search, k=3: 房间 里 有 一个 右手 拿 着 夹子 的 女人 在 给 一个 戴着 眼镜 的 男人 夹菜<br>Beam Search, k=5: 房间 里 有 一个 右手 拿 着 夹子 的 女人 在 给 一个 戴着 眼镜 的 男人 夹菜<br>Beam Search, k=20: 明亮 的 房间 里 坐 着 两位 在 干活 的 男士 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/4_bs_image.png) | Normal Max search: 一个 穿着 红色 衣服 的 女人 在 人群 前 的 T台 上 走秀<br>Beam Search, k=3: 一群 人 的 前面 有 一个 右手 拎 着 包 的 女人 在 T台 上 走秀<br>Beam Search, k=5: 平坦 的 T台 上 有 一位 双手 插 兜 的 女士 在 走秀<br>Beam Search, k=20: 人来人往 的 T台 上 有 一个 右手 拎 着 包 的 女人 在 走秀 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/5_bs_image.png) | Normal Max search: 道路 上 一个 人 的 旁边 有 一个 戴着 帽子 的 男人 在 采访 一个 戴着 帽子 的 男人<br>Beam Search, k=3: 道路 上 有 一个 戴着 帽子 的 男人 和 一个 戴着 帽子 的 男人 在 交谈<br>Beam Search, k=5: 道路 上 一群 人 的 旁边 有 一个 右手 拿 着 话筒 的 男人 在 讲话<br>Beam Search, k=20: 人来人往 的 道路 上 有 一个 戴着 帽子 的 男人 在 接受 采访 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/6_bs_image.png) | Normal Max search: 大厅 里 一群 人 的 旁边 有 一个 穿着 白色 上衣 的 女人 在 下 国际象棋<br>Beam Search, k=3: 大厅 里 一群 人 旁边 有 一个 戴着 眼镜 的 女人 在 下 国际象棋<br>Beam Search, k=5: 屋子里 一群 人 旁边 有 一个 戴着 眼镜 的 女人 在 下 国际象棋<br>Beam Search, k=20: 明亮 的 大厅 里 一群 人旁 有 一位 戴着 眼镜 的 女士 在 下 国际象棋 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/7_bs_image.png) | Normal Max search: 一个 穿着 比基尼 的 女人 坐在 海边 的 沙滩 上<br>Beam Search, k=3: 海边 的 沙滩 上 坐 着 一个 双手 放在 腿 上 的 女人<br>Beam Search, k=5: 海边 的 沙滩 上 坐 着 一个 穿着 比基尼 的 女人<br>Beam Search, k=20: 金灿灿 的 沙滩 上 坐 着 一个 长 头发 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/8_bs_image.png) | Normal Max search: 一个 穿着 短袖 的 男人 在 房间 里 骑 自行车<br>Beam Search, k=3: 房间 里 有 一个 坐在 轮椅 上 的 女人 在 弹 吉他<br>Beam Search, k=5: 房间 里 有 一个 坐在 轮椅 上 的 女人 在 弹 吉他<br>Beam Search, k=20: 明亮 的 房间 里 有 一位 双手 拿 着 扫把 的 女孩 在 扫地 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/9_bs_image.png) | Normal Max search: 两个 穿着 球衣 的 男人 在 球场上 争抢 足球<br>Beam Search, k=3: 足球场 上 有 两个 穿着 不同 球衣 的 男人 在 抢球<br>Beam Search, k=5: 足球场 上 有 两个 穿着 不同 球衣 的 男人 在 抢球<br>Beam Search, k=20: 绿草 茵茵 的 足球场 上 有 两个 男 运动员 在 踢足球 |

