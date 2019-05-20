import torch


def compute_iou(box1, box2):
    """
    Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    :param box1: (tensor) bounding boxes, sized [N,4].
    :param box2: (tensor) bounding boxes, sized [M,4].
    :return: (tensor) iou, sized [N,M].
    """

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


if __name__ == '__main__':
    print("test of unsqueeze and expand/expand_as")
    m = torch.zeros(1)
    n = torch.zeros(2, 2, 3, 5)
    print("m %s" % str(m.size()))
    print("n %s" % str(n.size()))

    m = m.unsqueeze(1).expand(2, 2, 3)
    print("\nafter m.unsqueeze(1).expand(2, 2, 3), \nm %s" % str(m.size()))

    m = m.unsqueeze(-1).expand_as(n)
    print("\nthen m.unsqueeze(-1).expand_as(n), \nm %s" % str(m.size()))
