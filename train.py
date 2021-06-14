import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #multi gpu: "0,1"
print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

### my code ###
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone

from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.domainclassifier import DomainClassifier

from io_utils import model_dict, parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, cifar100_few_shot

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
       raise ValueError('Unknown optimization, please define by yourself')     

    if  params.dann: #True goes in
        model_domain = DomainClassifier()
        model_domain = model_domain.cuda()
        optimizer_domain = torch.optim.Adam(model_domain.parameters())
    
    max_acc = 0
    for epoch in range(start_epoch,stop_epoch):
        #print("params.dann", params.dann)
        if params.dann: #True goes in
            model_domain.train()
            model.train()
            model.train_loop_dann(epoch, start_epoch, stop_epoch, base_loader,  optimizer, optimizer_domain, model_domain, params.dann_link )
        else:
            model.train()
            model.train_loop(epoch, base_loader,  optimizer )

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if params.dann: #True goes in
            model_domain.eval()
            model.eval()
            acc = model.test_loop_dann(val_loader)
        else:
            model.eval()
            acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            outfile_dec = os.path.join(params.checkpoint_dir, 'best_model_decoder.tar')
            torch.save({'epoch':epoch, 'state':model.decoder.state_dict()}, outfile_dec)
            if  params.dann: #True goes in
                outfile = os.path.join(params.checkpoint_dir, 'best_model_domain.tar')
                torch.save({'epoch':epoch, 'state':model_domain.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            outfile_dec = os.path.join(params.checkpoint_dir, '{:d}_decoder.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.decoder.state_dict()}, outfile_dec)
            if  params.dann: #True goes in
                outfile = os.path.join(params.checkpoint_dir, '{:d}_domain.tar'.format(epoch))
                torch.save({'epoch':epoch, 'state':model_domain.state_dict()}, outfile)
        
    ### my code ###
    out_record_file = os.path.join(params.checkpoint_dir, 'record.csv')
    if params.task in ["cdfsl-multi"] and os.path.isfile(out_record_file):
        out_record_file = os.path.join(params.checkpoint_dir, 'record2.csv')
    with open(out_record_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(model.record_list)
    if params.method in ['protonet']:
        out_test_file = os.path.join(params.checkpoint_dir, 'record_test.csv')
        if params.task in ["cdfsl-multi"] and os.path.isfile(out_test_file):
            out_test_file = os.path.join(params.checkpoint_dir, 'record_test2.csv')
        with open(out_test_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(model.test_list)
    ### my code ###
    
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    task   = params.task

    image_size = 224
    optimization = 'Adam'

    base_loaders = []
    val_loaders = []

    if params.method in ['baseline'] :
        if task in ["fsl", "cdfsl-single"]:
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loaders.append(datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(None)
        else:
            miniImageNet_datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 16)
            cifar100_datamgr = cifar100_few_shot.SimpleDataManager(image_size, batch_size = 16)

            base_loaders.append(miniImageNet_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(None)
            base_loaders.append(cifar100_datamgr.get_data_loader( aug = params.train_aug ))
            val_loaders.append(None)

        model           = BaselineTrain( model_dict[params.model], params.num_classes)

    elif params.method in ['protonet']:
        n_query = max(1, int(8* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        #n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if task in ["fsl", "cdfsl-single"]:
            datamgr     = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            val_datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)

            base_loaders.append(datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(val_datamgr.get_data_loader(aug = False))

        else:
            mini_ImageNet_datamgr     = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            mini_ImageNet_val_datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)
            cifar100_datamgr          = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="train",  **train_few_shot_params)
            cifar100_val_datamgr      = cifar100_few_shot.SetDataManager(image_size, n_query = n_query, mode="val",  **test_few_shot_params)


            base_loaders.append(mini_ImageNet_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(mini_ImageNet_val_datamgr.get_data_loader(aug = False))
            base_loaders.append(cifar100_datamgr.get_data_loader(aug = params.train_aug ))
            val_loaders.append(cifar100_val_datamgr.get_data_loader(aug = False))

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    save_dir =  configs.save_dir

    task_path = 'single' if task in ["fsl", "cdfsl-single"] else 'multi'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, task_path, params.model, params.method)
    # Decoder
    params.checkpoint_dir += "_decoder"
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    
    print("params.dann", params.dann)
    if params.dann: #True goes in
        params.checkpoint_dir += '_dann_'
        params.checkpoint_dir += params.dann_link
        
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if task in ["fsl", "cdfsl-single"]:
        model = train(base_loaders[0], val_loaders[0], model, optimization, start_epoch, stop_epoch, params)
    ### my code ###
    elif params.dann: #True goes in
        model = train(base_loaders, val_loaders, model, optimization, start_epoch, stop_epoch, params)
    ### my code ###
    else:
        stop_epoch = stop_epoch // 2
        for i in range(2):
            print('now source domain ', i, ":")
            model = train(base_loaders[i], val_loaders[i], model, optimization, start_epoch, stop_epoch, params)
