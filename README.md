## 环境
OS: x64 Linux 4.19
python: Anaconda3, python3.6
需要额外安装的第三方库版本信息：
```
matplotlib (3.0.2)
MedPy (0.4.0)
nibabel (2.1.0)
numpy (1.14.2)
opencv-python (4.0.0.21)
pandas (0.24.1)
Pillow (5.4.1)
psutil (5.5.1)
scikit-image (0.14.2)
scikit-learn (0.20.2)
scipy (1.2.1)
SimpleITK (1.2.0)
tensorboard (1.8.0)
tensorflow (1.8.0)
```
请使用Anaconda安装tensorflow及其依赖，如机器带Nvidia GPU，请安装tensorflow-gpu=1.8.0,
否则请安装tensorflow-cpu=1.8.0，Anaconda会自动安装CUDA,CUDNN，MKL等依赖
其余第三方库请使用pip安装: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 库名(上面对应库名，所有大写改为小写)==版本号


## preprocess.py:
* 制作mask并保存
* 原图截断到[-200,250]后乘以mask，生成新的图像供训练和测试使用
* 在mask中找到bbox保存到pickle文件中供训练和测试使用

(OTSU的opencv实现)[https://segmentfault.com/a/1190000015647247]
(膨胀、腐蚀、开运算、闭运算的opencv实现)[https://segmentfault.com/a/1190000015650320]


## dataset.py
* random_batch
随机提供数据,在ROI中随机截取适当大的部分(224 * 224 * 24)提供给训练器，所有label=0的图像中提取的bbox都是正常，
所有label=1的图像中提取的bbox都认为是肿瘤(容易造成假阳性，待解决)

* test_batch
滑动窗口提供同样大的部分(224 * 224 * 24)

## model.py
* 基于patch训练
结构： 3D Residual Network + Fully Connected Layer 
输入： x: 单个patch(224 * 224 * 24), y: 单个patch的标签，暂时和整个图像的标签一致
输出： {0,1}

* 基于整个3D图像训练(解决假阳性问题)
结构： 随机提取若干个patch + 基于patch训练 + (阈值？投票？)
输入： x: 整个CT图像(512 * 512 * depth), y: 整个CT图像的标签
输出： {0,1}


## train.py



## test.py
