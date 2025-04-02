#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util.py
@Time: 4/5/19 3:47 PM
"""

import numpy as np
import torch
import torch.nn.functional as F

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss

# 선택 사항: 만약 PointNet의 feature transform regularizer를 사용하고 싶다면,
# 아래의 함수를 pointnet_utils.py에서 가져와서 사용할 수 있습니다.
# def feature_transform_reguliarzer(trans):
#     d = trans.size()[1]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.cuda()
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
#     return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
