import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data

import cv2


class YOLODataset(data.Dataset):
    image_size = 224

    def __init__(self, root, list_file, train, transform):
        print("数据初始化")
        self.root = root
        self.train = train
        self.transform = transform  # 对图像转化
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)

        with open(list_file) as f:
            lines = f.readlines()

        # 遍历voc2012train.txt的每一行
        # name num_faces x y x y c ......
        # 转换后，每个index对应一组 box label
        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_faces = int(splited[1])  # 一张图像中真实框的总数
            box = []
            label = []
            for i in range(num_faces):
                x = float(splited[2 + 5 * i])
                y = float(splited[3 + 5 * i])
                x2 = float(splited[4 + 5 * i])
                y2 = float(splited[5 + 5 * i])
                c = splited[6 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)  # 数据集中图像的总数

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        img = cv2.imread(os.path.join(self.root + fname))
        # clone 深复制，不共享内存
        # 拿出对应的bbox及 标签对应的序号
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # Todo
        # 如果为训练集,进行数据增强
        # if self.train:
        #     img, boxes = self.random_flip(img, boxes)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # x, y, w, h, 除以图片的长宽
        img = self.BGR2RGB(img)
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels)  # 7x7x30
        for t in self.transform:  # 图像转化
            img = t(img)
        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        """
        将每张图片对应的box信息编码为 （7，7，30）的target Tensor
        :param boxes: (tensor) [[x1,y1,x2,y2],[x1,y1,x2,y2],[]]，注意，数值已经被规格化了，数值在0-1之间。
        :param labels: (tensor) [...]
        :return: 7x7x30，x,y转化为box中心到cell左上顶点的相对位置。w,h的取值相对于整幅图像的尺寸
        """
        target = torch.zeros((7, 7, 30))
        cell_size = 1. / 7
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):  # cxcy.size()[0]是框的数量
            cxcy_sample = cxcy[i]  # 取其中一个box
            # cxcy_sample为一个物体的中心点坐标，求该坐标位于7x7网格的哪个网格
            # cxcy_sample坐标在0-1之间  现在求它再0-7之间的值，故乘以7
            # ij长度为2，代表7x7个cell中的某个cell，负责预测一个物体
            ij = (cxcy_sample / cell_size).ceil() - 1  # ceil 向上取整
            # 每行的第4和第9的值设置为1，即每个网格提供的两个真实候选框 框住物体的概率是1.
            # xml中坐标理解：原图像左上角为原点，右边为x轴，下边为y轴。
            # 而二维矩阵（x，y）  x代表第几行，y代表第几列
            # 假设ij为（1,2） 代表x轴方向长度为1，y轴方向长度为2
            # 二维矩阵取（2,1） 从0开始，代表第2行，第1列的值
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            # cxcy_sample：第i个bbox的中心点坐标
            # cell_lt_xy：对应cell的左上角相对坐标（相对于图片长宽（224*224）的比例，0-1之间）
            # delta_xy：真实框的中心点坐标相对于该中心点所在cell左上角的相对坐标。
            # cxcy_sample - cell_lt_xy肯定小于1/7，否则box与cell不对应，故*7，扩展到0-1，猜测是为了便于收敛。
            cell_lt_xy = ij * cell_size
            delta_xy = (cxcy_sample - cell_lt_xy) / cell_size

            # x,y代表了检测框中心相对于cell边框的坐标。w,h的取值相对于整幅图像的尺寸
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        '''
        随机翻转
        '''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes
