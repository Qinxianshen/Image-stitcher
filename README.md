# Image-stitcher

### (1)使用Opencv里的特征点提取计算单应矩阵（3*3）进行图片拼接

##### 原图

![原图](./pic/1.jpg)

![原图](./pic/2.jpg)

##### 特征点提取

![特征点提取](./pic/sift_keypoints_1.jpg)

#### 拼接结果

![拼接结果](./pic/result_1.jpg)

### (2)使用CNN 来提取特征点来计算单应矩阵 进行图片拼接

![网络概括](./pic/1.png)


![网络概括](./pic/2.png)


#### 原始图片

![网络概括](./pic/3.png)

#### 变化图片

![网络概括](./pic/4.png)


### (3)使用无监督的方法

![网络概括](./pic/unsuper_hnet.png)

#### 评估方法

![评估方法](./pic/hnet_pinggu.png)

![评估方法](./pic/hnet_pinggu2.png)


### (4) 计算误差的方法： MACE (Mean Ave. Corner Error)

200次 :('Ave. Corner Error: ', '270.4477227329222')

1000次：('Ave. Corner Error: ', '73.6007446534849')




### 文件功能：

（1）change_test.py  编写了get_test2 方便之后计算两张图片的H矩阵

（2）change_test.py 用于测试并计算MACE

（3）Hnet_train.py 用于训练




#### Tips

[Homography Net](https://arxiv.org/pdf/1606.03798.pdf)

[Homography Net](https://blog.csdn.net/ajing2014/article/details/53998866)
