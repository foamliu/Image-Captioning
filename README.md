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
下载 [预训练模型](https://github.com/foamliu/Image-Captioning/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录，然后执行:

```bash
$ python demo.py
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/2_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/3_image.png) |
|马场 上 有 一个 戴着 头盔 的 男人 在 骑马 | 一个 穿着 黑色 上衣 的 男人 搂 着 一个 穿着 裙子 的 女人 站 在 道路 上 | 大厅 里 一个 人 的 旁边 有 一个 穿着 短袖 的 男人 在 帮 一个 穿着 短袖 的 男人 穿着 鞋子 | 一个 人 的 前面 有 一个 穿着 球衣 的 男人 在 运动场 上 踢足球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/6_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/7_image.png) |
|一个 穿着 黑色 上衣 的 男人 和 一个 穿着 运动服 的 男人 站 在 运动场 上 | 一个 穿着 裙子 的 女人 在 舞台 上 跳舞 | 一个 穿着 黑色 衣服 的 男人 在 屋子 外 的 道路 上 骑 自行车 | 两个 穿着 球衣 的 男人 在 球场上 踢足球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/9_image.png) |![image](https://github.com/foamliu/Image-Captioning/raw/master/images/10_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/11_image.png)|
|一个 人 的 前面 有 一个 穿着 黑色 衣服 的 男人 和 一个 穿着 黑色 上衣 的 男人 在 大厅 里 交流 | 舞台 上 有 两个 穿着 黑色 衣服 的 男人 在 表演 |白雪皑皑 的 雪地 上 有 一个 戴着 帽子 的 人 在 滑雪 | 一个 穿着 运动服 的 男人 在 运动场 上 打网球 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/12_image.png)  | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/13_image.png) |![image](https://github.com/foamliu/Image-Captioning/raw/master/images/14_image.png)| ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/15_image.png)|
|一个 穿着 黑色 衣服 的 男人 在 一望无际 的 海里 骑马|一个 穿着 运动服 的 男人 在 运动场 上 踢足球 | 一个 穿着 运动服 的 女人 在 运动场 上 打乒乓球 | 一个 穿着 黑色 上衣 的 男人 在 果园 里 穿着 各异 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/16_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/17_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/18_image.png) | ![image](https://github.com/foamliu/Image-Captioning/raw/master/images/19_image.png) |
|一个 穿着 短袖 的 男人 在 道路 上 骑 自行车 | 一个 右手 拿 着 话筒 的 男人 在 舞台 上 唱歌 | 两个 穿着 球衣 的 男人 在 球场上 争抢 足球 | 两个 穿着 球衣 的 男人 在 球场上 争抢 足球 |

#### 光束搜索 (Beam Search)
1 | 2 |
|---|---|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/0_bs_image.png) | Normal Max search: 一个 戴着 墨镜 的 男人 和 一个 穿着 黑色 外套 的 女人 走 在 大厅 里 <br> Beam Search, k=3: 道路 上 两个 人 的 旁边 走 着 一个 右手 拿 着 手机 的 女人 <br> Beam Search, k=5: 大厅 里 两个 人 的 旁边 走 着 一个 右手 拿 着 手机 的 女人 <br> Beam Search, k=7: 道路 上 三个 人 的 旁边 走 着 一个 右手 拿 着 手机 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/1_bs_image.png) | Normal Max search: 一个 穿着 白色 上衣 的 男人 和 一个 穿着 裙子 的 女人 站 在 室外 的 栏杆 旁 <br> Beam Search, k=3: 一个 戴着 帽子 的 男人 和 一个 右手 拿 着 东西 的 女人 站 在 道路 上 <br> Beam Search, k=5: 道路 上 有 一个 右手 拿 着 话筒 的 女人 在 采访 一个 右手 拿 着 东西 的 女人 <br> Beam Search, k=7: 道路 上 站 着 两个 手里 拿 着 话筒 的 人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/2_bs_image.png) | Normal Max search: 一个 穿着 裙子 的 女人 和 一个 穿着 西装 的 男人 走 在 道路 上 <br> Beam Search, k=3: 道路 上 两个 人 的 旁边 走 着 一个 右手 拿 着 包 的 女人 <br> Beam Search, k=5: 道路 上 走 着 一个 右手 拿 着 手机 的 女人 和 一个 右手 拿 着 手机 的 女人 <br> Beam Search, k=7: 道路 上 走 着 一个 右手 拿 着 东西 的 女人 |
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/3_bs_image.png) | Normal Max search: 两个 穿着 球衣 的 男人 在 球场上 争抢 足球 <br> Beam Search, k=3: 两个 穿着 运动服 的 男人 在 运动场 上 争抢 足球<br>Beam Search, k=5: 足球场 上 有 两个 穿着 运动服 的 男人 在 争抢 足球<br>Beam Search, k=7: 碧绿 的 球场上 有 两位 穿着 球服 的 男士 在 抢 足球|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/4_bs_image.png) | Normal Max search: 一个 穿着 运动服 的 男人 在 运动场 上 踢足球<br>Beam Search, k=3: 运动场 上 站 着 一个 双手 拿 着 奖杯 的 男人<br>Beam Search, k=5: 绿油油 的 球场上 有 一个 穿着 球衣 的 男人 在 踢足球<br>Beam Search, k=7: 碧绿 的 球场上 站 着 一位 穿着 球服 的 男士|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/5_bs_image.png) | Normal Max search: 两个 穿着 球衣 的 男人 在 球场上 踢足球<br>Beam Search, k=3: 两个 穿着 球衣 的 人 在 宽阔 的 运动场 上 争抢 足球<br>Beam Search, k=5: 两个 穿着 不同 球衣 的 男人 在 运动场 上 争抢 足球<br>Beam Search, k=7: 碧绿 的 球场上 有 两位 穿着 球服 的 男士 在 抢 足球|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/6_bs_image.png) | Normal Max search: 一个 穿着 黑色 衣服 的 男人 和 一个 穿着 白色 衣服 的 女人 站 在 室内 的 桌子 旁<br>Beam Search, k=3: 房间 里 有 一个 右手 拿 着 话筒 的 男人 在 采访 一个 双手 放在 身前 的 男人<br>Beam Search, k=5: 房间 里 三个 人 的 旁边 有 一个 右手 拿 着 笔 的 男人 在 写字<br>Beam Search, k=7: 房间 里 有 一个 右手 拿 着 话筒 的 女人 在 采访 一个 双手 放在 身前 的 男人|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/7_bs_image.png) | Normal Max search: 一个 穿着 西装 的 男人 和 一个 穿着 裙子 的 女人 站 在 舞台 上<br>Beam Search, k=3: 一个 穿着 西装 的 男人 搂 着 一个 右手 拿 着 包 的 女人 站 在 展板 前<br>Beam Search, k=5: 舞台 上 站 着 两个 手里 拿 着 话筒 的 人<br>Beam Search, k=7: 舞台 上 有 一个 右手 拿 着 话筒 的 男人 在 讲话|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/8_bs_image.png) | Normal Max search: 房间 里 有 一个 穿着 白色 上衣 的 女人 在 给 一个 穿着 白色 上衣 的 女人 递 东西<br>Beam Search, k=3: 房间 里 有 一个 右手 拿 着 筷子 的 女人 在 喂 一个 孩子 吃 东西<br>Beam Search, k=5: 房间 里 有 一个 右手 拿 着 筷子 的 女人 在 喂 一个 孩子 吃 东西<br>Beam Search, k=7: 房间 里 有 一个 扎 着 头发 的 女人 和 一个 右手 拿 着 筷子 的 女人 在 吃饭|
|![image](https://github.com/foamliu/Image-Captioning/raw/master/images/9_bs_image.png) | Normal Max search: 一个 穿着 黑色 衣服 的 男人 和 一个 穿着 黑色 衣服 的 男人 站 在 房屋 外 的 道路 上<br>Beam Search, k=3: 室外 有 一个 戴着 帽子 的 男人 和 一个 戴着 帽子 的 男人 在 交谈<br>Beam Search, k=5: 道路 上 站 着 一个 双手 拿 着 东西 的 男人<br>Beam Search, k=7: 道路 上 站 着 一个 双手 拿 着 东西 的 男人|
