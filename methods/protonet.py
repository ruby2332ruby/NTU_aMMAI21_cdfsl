# This code is modified from https://github.com/jakesnell/prototypical-networks 
# Decoder part is modified from https://github.com/arnaghosh/Auto-Encoder/blob/master/resnet.py

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

import utils

class Decoder(nn.Module):
    def __init__(self, zsize):
        super(Decoder,self).__init__()
        self.dfc3 = nn.Linear(zsize, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096,256 * 6 * 6)
        self.bn1 = nn.BatchNorm1d(256*6*6)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding = 0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding = 1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding = 1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding = 2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride = 4, padding = 4)

    def forward(self,x):#,i1,i2,i3):
        
        x = self.dfc3(x)
        #x = F.relu(x)
        x = F.relu(self.bn3(x))
        
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        #x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        #x = F.relu(x)
        #print(x.size())
        x = x.view(-1,256,6,6)
        #print (x.size())
        x=self.upsample1(x)
        #print x.size()
        x = self.dconv5(x)
        #print x.size()
        x = F.relu(x)
        #print x.size()
        x = F.relu(self.dconv4(x))
        #print x.size()
        x = F.relu(self.dconv3(x))
        #print x.size()		
        x=self.upsample1(x)
        #print x.size()		
        x = self.dconv2(x)
        #print x.size()		
        x = F.relu(x)
        x=self.upsample1(x)
        #print x.size()
        x = self.dconv1(x)
        #print x.size()
        x = torch.sigmoid(x)
        return x

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn  = nn.CrossEntropyLoss()
        self.decoder = Decoder(self.feature.final_feat_dim)
        self.autoencoder_criterion = nn.MSELoss()


    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )


        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        return scores
    
    def set_loss_decoder(self, x):
        x     = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        z_all = self.feature.forward(x)
        x_hat = self.decoder.forward(z_all)
        loss  = self.autoencoder_criterion(x_hat, x)
        return loss


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        # scores = self.set_forward(x)
        # loss = self.loss_fn(scores, y_query)
        loss = self.set_loss_decoder(x)

        return loss

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2) # N x M
