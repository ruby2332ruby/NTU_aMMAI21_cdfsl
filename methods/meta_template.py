import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True, device='cuda:0'):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test
        self.device     = device
        
        ### my code ###
        self.record_list = [["Epoch", "Batch", "Loss"]]
        self.test_list = [["iter_num", "acc_mean", "+/-"]]
    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.to(self.device))
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):       
        scores  = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = 10

        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                
                ### my code ###
                self.record_list.append([epoch, i, avg_loss/float(i+1)])

    def test_loop(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        ### my code ###
        self.test_list.append([iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)])

        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.to(self.device))

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.to(self.device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(self.device)
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).to(self.device)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

    ### my code ###
    #model.train_loop_dann(epoch, base_loader,  optimizer, optimizer_domain, model_domain )
    def train_loop_dann(self, epoch, train_loader, optimizer, optimizer_domain, model_domain):
        print_freq = 10
        avg_loss=0
        avg_loss_domain=0
        lamb = 0.7
        self.record_list = [["Epoch", "Batch", "Loss", "Domain Loss"]]
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