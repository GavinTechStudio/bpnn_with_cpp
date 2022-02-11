# BP神经网络的C++实现

视频地址：[彻底搞懂BP神经网络 理论推导+代码实现（C++）](https://www.bilibili.com/video/BV1Y64y1z7jM)

### 注意

本项目代码已重构，当前代码不再建议使用。

重构代码项目
[![GavinTechStudio/Back-Propagation-Neural-Network - GitHub](https://gh-card.dev/repos/GavinTechStudio/Back-Propagation-Neural-Network.svg)](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network)

### 可能遇到的问题

#### 收敛问题

项目中提供的`traindata.txt`和`testdata.txt`换行符为`\r`，所以可能在Windows环境下会出现读入数据多一个空行，导致程序无法收敛。

[bp网络无法收敛 · Issue #1 · GavinTechStudio/bpnn_with_cpp (github.com)](https://github.com/GavinTechStudio/bpnn_with_cpp/issues/1)

#### 数据读入问题

如遇到输出`Error in reading traindata.txt`，则是因为`traindata.txt`和`testdata.txt`所放位置不正确导致的，应将这两个数据文件放到对应的可执行程序的目录下。
