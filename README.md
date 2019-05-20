# Easy_Pytorch_YOLOv1
施工中

## IOU
[![EOF7vt.png](https://s2.ax1x.com/2019/05/18/EOF7vt.png)](https://imgchr.com/i/EOF7vt)

由于坐标轴的设定，Y轴数值向下为增加，所以
$$
intersection = [min(y2, y2') - max(y1, y1')] * [min(x2, x2') - max(x1, x1')]
$$

## loss
$$
\begin{split}
loss &= \lambda_{\text { coord }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text { obj }}\left[\left(x_{i}-\hat{x}_{i}\right)^{2}+\left(y_{i}-\hat{y}_{i}\right)^{2}\right]\\

&+\lambda_{\text { coord }} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\text { obj } j}\left[\left(\sqrt{w_{i}}-\sqrt{\hat{w}_{i}}\right)^{2}+\left(\sqrt{h_{i}}-\sqrt{\hat{h}_{i}}\right)^{2}\right]\\

&+\sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\mathrm{obj}}\left(C_{i}-\hat{C}_{i}\right)^{2}\\

&+\lambda_{\mathrm{noobj}} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{i j}^{\mathrm{noobj}}\left(C_{i}-\hat{C}_{i}\right)^{2}\\

&+\sum_{i=0}^{S^{2}} \mathbb{1}_{i}^{\mathrm{obj}} \sum_{c \in \text { classes }}\left(p_{i}(c)-\hat{p}_{i}(c)\right)^{2}\\

\end{split}
$$

loss.py文件把loss分成几部分，则有四种情况：

1 target存在框，prediction没有框：计算not response loss 未响应损失
2 target存在框，prediction存在框：计算response loss响应损失
3 target没有框，prediction没有框：无损失(不计算)
4 target没有框，prediction存在框：计算不包含obj损失  只计算confidence loss

## Data
voc的数据格式是xml，转换为txt，转换代码在data目录下。定义YOLODataset。

## Todo
- [ ] 训练
- [ ] 评估函数



