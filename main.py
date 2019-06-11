import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import cv2

from model.yolo import YOLO
from util.visualize import Visualizer
from config import conf
from data.dataset import YOLODataset
from model.loss import YOLOLoss


def train():
    # vis = Visualizer(conf.env)

    print("==============================\n网络结构  开始 \n==============================")
    yolo = YOLO()
    print("----------\n网络结构  加载预训练模型\n----------")
    if conf.load_model_path:
        yolo.load_state_dict(torch.load(conf.load_model_path, map_location=lambda storage, loc: storage))
    if conf.use_gpu:
        yolo.cuda()
    print("----------\n网络结构  输出网络\n----------")
    print(yolo)
    yolo.train()
    print("==============================\n网络结构  结束 \n==============================")

    print("==============================\n加载数据  开始 \n==============================")
    train_dataset = YOLODataset(root=conf.file_root, list_file=conf.voc_2012train, train=True,
                                transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    test_dataset = YOLODataset(root=conf.test_root, list_file=conf.voc_2007test, train=False,
                               transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    print("==============================\n加载数据  结束 \n==============================")

    criterion = YOLOLoss(7, 2, 5, 0.5)
    optimizer = torch.optim.SGD(yolo.parameters(), lr=conf.learning_rate, momentum=conf.momentum,
                                weight_decay=conf.weight_decay)
    logfile = open('log/log.txt', 'w')
    best_test_loss = np.inf
    print('训练集有 %d 张图像' % (len(train_dataset)))
    print('一个batch的大小为 %d' % (conf.batch_size))

    print("==============================\n训练  开始 \n==============================")
    for epoch in range(conf.num_epochs):
        if epoch == 1:
            conf.learning_rate = 0.0005
        if epoch == 2:
            conf.learning_rate = 0.00075
        if epoch == 3:
            conf.learning_rate = 0.001
        if epoch == 80:
            conf.learning_rate = 0.0001
        if epoch == 100:
            conf.learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = conf.learning_rate
            # 第几次epoch  及 当前epoch的学习率
        print('\n\n当前的epoch为 %d / %d' % (epoch + 1, conf.num_epochs))
        print('当前epoch的学习率: {}'.format(conf.learning_rate))

        total_loss = 0.
        for i, (images, target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            if conf.use_gpu:
                images, target = images.cuda(), target.cuda()
            pred = yolo(images)
            loss = criterion(pred, target)
            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % conf.print_freq == 0:
                print('在训练集上：当前epoch为 [%d/%d], Iter [%d/%d] 当前batch损失为: %.4f, 当前epoch到目前为止平均损失为: %.4f'
                      % (epoch + 1, conf.num_epochs, i + 1, len(train_loader), loss.data, total_loss / (i + 1)))
                # 画出训练集的平均损失
                # vis.plot_train_val(loss_train=total_loss / (i + 1))
        torch.save(yolo.state_dict(), conf.current_epoch_model_path)

        print("----------\n训练  对当前train做验证\n----------")
        validation_loss = 0.0
        yolo.eval()
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images, volatile=True)  # volatile标记. 只能通过keyword传入.
            target = Variable(target, volatile=True)
            if conf.use_gpu:
                images, target = images.cuda(), target.cuda()
            pred = yolo(images)
            loss = criterion(pred, target)
            validation_loss += loss.data[0]
        validation_loss /= len(test_loader)
        # vis.plot_train_val(loss_val=validation_loss)
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('当前得到最好的验证集的平均损失为  %.5f' % best_test_loss)
            torch.save(yolo.state_dict(), conf.best_test_loss_model_path)

        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()


def predict():
    pass


def eval():
    pass


if __name__ == '__main__':
    import fire

    fire.Fire()

    train()
