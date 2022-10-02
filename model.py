#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/10/2 上午11:40
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : model.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>


from torch import nn
from torch.nn import functional as F

class Contrastive_Model(nn.Module):
    def __init__(self, topic_model=None, contextual_model=None, device='cuda:0'):
        super(Contrastive_Model, self).__init__()
        self.topic_model = topic_model
        self.contextual_model = contextual_model
        self.contextual_embedding = nn.Linear(768, 100)
        self.bn_layer = nn.BatchNorm1d(100)
        # self.softmax = F.softmax()
        self.device = device

    def encode(self, doc_embedding):
        # doc_embedding = self.contextual_model.encode(doc)
        # doc_embedding = torch.from_numpy(doc_embedding).to(self.device)
        z = self.contextual_embedding(doc_embedding)
        # return F.softmax(z, dim=-1)
        return F.relu(self.bn_layer(z))
        # return z

    def forward(self, bow, doc):
        z = self.encode(doc)

        theta = self.topic_model.get_theta(bow)

        return z, theta