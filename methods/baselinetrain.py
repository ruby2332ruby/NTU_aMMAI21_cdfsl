import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from math import exp

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax'):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        
        ### my code ###
        self.record_list = [["Epoch", "Batch", "Loss", "Top1 Val", "Top1 Avg"]]

    def forward(self,x):
        x    = Variable(x.cuda())
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y )
    
    def train_loop(self, epoch, start_epoch, stop_epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+loss.item()
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1), self.top1.val, self.top1.avg))
                
                ### my code ###
                self.record_list.append([epoch, i, avg_loss/float(i+1), self.top1.val, self.top1.avg])
                     
    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration


    ### my code ###
    #model.train_loop_dann(epoch, base_loader,  optimizer, optimizer_domain, model_domain )
    def train_loop_dann(self, epoch, start_epoch, stop_epoch, train_loader, optimizer, optimizer_domain, model_domain, dann_link):
        print_freq = 10
        avg_loss=0
        avg_loss_domain=0
        gam = 10
        progress = epoch/(stop_epoch-start_epoch-1)
        lamb = (2 / (1+exp(-gam*progress))) -1
        self.record_list = [["Epoch", "Batch", "Loss", "Domain Loss", "Top1 Val", "Top1 Avg"]]
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(train_loader[0], train_loader[1])):
            x1 = Variable(x1.cuda())
            y1 = Variable(y1.cuda())
            x2 = Variable(x2.cuda())
            y2 = Variable(y2.cuda())
            
            optimizer.zero_grad()
            optimizer_domain.zero_grad()
            
            # step1: train domain classifier
            mixed_data, mixed_label, domain_label = model_domain.mix_data(x1, x2, y1, y2)
            feature = self.feature.forward(mixed_data)
            # We don't need to train feature extractor in step 1.
            # Thus we detach the feature neuron to avoid backpropgation.
            domain_logits = model_domain.forward(feature.detach())
            loss = model_domain.loss_fn(domain_logits, domain_label)
            avg_loss_domain = avg_loss_domain+loss.item()
            loss.backward()
            optimizer_domain.step()
            
            # step 2 : train feature extractor and label classifier
            class_logits = self.classifier.forward(feature)
            _, predicted = torch.max(class_logits.data, 1)
            correct = predicted.eq(mixed_label.data).cpu().sum()
            self.top1.update(correct.item()*100 / (mixed_label.size(0)+0.0), mixed_label.size(0))
            domain_logits = model_domain.forward(feature)
            # loss = cross entropy of classification - lamb * domain binary cross entropy.
            #  The reason why using subtraction is similar to generator loss in disciminator of GAN
            loss = self.loss_fn(class_logits, mixed_label) - lamb * model_domain.loss_fn(domain_logits, domain_label)
            avg_loss = avg_loss+loss.item()
            loss.backward()
            optimizer.step()
            
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                batch_num_mul = min(len(train_loader[0]), len(train_loader[1]))
                batch_num_mul = batch_num_mul//10 *10
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Domain Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i, batch_num_mul, avg_loss/float(i+1), avg_loss_domain/float(i+1), self.top1.val, self.top1.avg))
                
                self.record_list.append([epoch, i, avg_loss/float(i+1), avg_loss_domain/float(i+1), self.top1.val, self.top1.avg])
                
    def test_loop_dann(self, val_loader):
        return -1 #no validation, just save model during iteration