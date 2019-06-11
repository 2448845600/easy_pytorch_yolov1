import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from torch.autograd import Variable

from util.iou import compute_iou


class YOLOLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, predict, target):
        """
        参考README.md
        :param predict: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        :param target: (tensor) size(batchsize,S,S,30)
        :return: loss
        """

        N = predict.size()[0]
        coord_mask = target[:, :, :, 4] > 0

        # Q: why not noobj_mask = target[:, :, :, 4] == 0 and target[:, :, :, 9] == 0
        # A: target is organize each box, the second box is empty or meaningless.
        noobj_mask = target[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        # get coord prediction and target
        coord_pred = predict[coord_mask].view(-1, 30)
        coord_box_pred = coord_pred[:, :10].contiguous().view(-1, 5)
        coord_class_pred = coord_pred[:, 10:]

        coord_target = target[coord_mask].view(-1, 30)
        coord_box_target = coord_target[:, :10].contiguous().view(-1, 5)
        coord_class_target = coord_target[:, 10:]

        # get no object prediction and target, and compute noobj_loss
        noobj_pred = predict[noobj_mask].view(-1, 30)
        noobj_target = target[noobj_mask].view(-1, 30)
        noobj_pred_mask = torch.cuda.ByteTensor(noobj_pred.size())
        noobj_pred_mask.zero_()
        noobj_pred_mask[:, 4] = 1
        noobj_pred_mask[:, 9] = 1
        noobj_pred_c = noobj_pred[noobj_pred_mask]
        noobj_target_c = noobj_target[noobj_pred_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_target_c, size_average=False)

        # compute obj loss
        coord_response_mask = torch.cuda.ByteTensor(coord_box_target.size())
        coord_response_mask.zero_()
        coord_not_response_mask = torch.cuda.ByteTensor(coord_box_target.size())
        coord_not_response_mask.zero_()
        box_target_iou = torch.zeros(coord_box_target.size()).cuda()

        for i in range(0, coord_box_target.size()[0], 2):
            # change data format from xywh to xyxy
            # box1 means two boxes predicted in one cell
            # box2 means all target boxes
            # now, let's search best IOU box
            box1 = coord_box_pred[i:i + 2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2] - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] + 0.5 * box1[:, 2:4]
            box2 = coord_box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] + 0.5 * box2[:, 2:4]

            # iou = compute_iou(box1_xyxy, box2_xyxy)  # iou: tensor([2, 1])
            iou = nms(box1_xyxy, box2_xyxy) # use torchvision0.3
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()
            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + 1 - max_index] = 1
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).date.cuda()
        box_target_iou = Variable(box_target_iou).cuda()

        # response loss
        coord_box_pred_response = coord_box_pred[coord_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coord_response_mask].view(-1, 5)  # 计算confid loss时，target confid都是1
        box_target_response = coord_box_target[coord_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(coord_box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        loc_loss = F.mse_loss(coord_box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + \
                   F.mse_loss(torch.sqrt(coord_box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]),
                              size_average=False)

        # not response loss
        coord_box_pred_not_response = coord_box_pred[coord_not_response_mask].view(-1, 5)
        box_target_not_response = coord_box_target[coord_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        not_contain_loss = F.mse_loss(coord_box_pred_not_response[:, 4], box_target_not_response[:, 4],
                                      size_average=False)

        # class loss
        class_loss = F.mse_loss(coord_class_pred, coord_class_target, size_average=False)

        loss = (self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss
                + self.l_noobj * noobj_loss + class_loss) / N

        return loss


if __name__ == '__main__':
    m = torch.rand(2, 2, 2)
    # n = torch.arange(8).view(2, 2, 2)
    # print(m)
    # print(n)
    #
    # flag = n < 3
    # print(flag)
    # k = m[flag].view(-1, 3)
    # print(k)

    m = torch.rand(2, 4)
    print(m)
    n = m.view(-1, 2)
    print(n)
