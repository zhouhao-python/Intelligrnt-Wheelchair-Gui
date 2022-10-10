## 智能眼动轮椅控制平台
* author:周浩
* 日期:2022/4/25
---
### 原理
<pre>前级摄像头采集图像，TX2出来每帧图像、提取人脸、人眼、送入MEANet，输出结果发送Arduino、Arduino控制舵机协同作用，完成轮椅运动。</pre>

---
### 图标案例
![人眼](./image/preup.png)

---
### 网络结构
![MEANet](./image/MEANet.png)

---
### 实验结果
![验证矩阵](./image/result.png)