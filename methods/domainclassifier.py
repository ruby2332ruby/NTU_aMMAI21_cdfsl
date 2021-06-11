# This code is modified from https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW11/HW11_ZH.ipynb and baselinetrain.py
import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, h):
        h = Variable(h.cuda())
        y = self.layer(h)
        return y
    
    def mix_data(self, x1, x2, y1, y2):
        
        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([x1, x2], dim=0)
        mixed_label = torch.cat([y1, y2], dim=0)
        domain_label = Variable(torch.zeros([x1.shape[0] + x2.shape[0], 1]).cuda())
        # set domain label of source data to be 1.
        domain_label[:x1.shape[0]] = 1
        return mixed_data, mixed_label, domain_label
