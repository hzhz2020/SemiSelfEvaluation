#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
naming convention:

[labeledtrain, labeledtrain_batchsize, labeledtrain_iter, labeledtrain_loader, labeledtrain_dataset]

[unlabeledtrain, unlabeledtrain_batchsize, unlabeledtrain_iter, unlabeledtrain_loader, unlabeledtrain_dataset]

'''
import argparse
import logging
import math
import os
import random
import shutil
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from libml.TissueMNIST_data import TissueMNIST as dataset
from libml.utils import save_pickle
from libml.utils import train_one_epoch, eval_model
from libml.utils import EarlyStopping

from torchvision import models
from libml.models.ema import ModelEMA

from libml.utils.pseudo_label import PL

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
#experiment setting
parser.add_argument('--dataset_name', default='TissueMNIST', type=str, help='name of dataset')
parser.add_argument('--data_seed', default=0, type=int, help='random seed data partitioning procedure')
parser.add_argument('--training_seed', default=0, type=int, help='random seed for training procedure')
parser.add_argument("--nlabels", "-n", default=80, type=int, help="the number of labeled data")

parser.add_argument('--arch', default='resnet18', type=str, help='backbone to use')
parser.add_argument('--train_epoch', default=300, type=int, help='total epochs to run')
# parser.add_argument('--nimg_per_epoch', default=2400, type=int, help='how many images in the labeled train set')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--eval_every_Xepoch', default=1, type=int, help='manual epoch number (useful on restarts)')


#hyperparameters inherit from Echo_ClinicalManualScript_torch style
parser.add_argument('--resume', default='', type=str,
                    help='name of the checkpoint (default: none)')

parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
                    help='fullpath of the checkpoint to resume from(default: none)')

parser.add_argument('--train_dir', default='/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_1112/experiments/ViewClassifier/seed0/DEV479',
                    help='directory to output the result')


#data paths
parser.add_argument('--l_train_dataset_path', default='', type=str)
parser.add_argument('--u_train_dataset_path', default='', type=str)
parser.add_argument('--val_dataset_path', default='', type=str)
parser.add_argument('--test_dataset_path', default='', type=str)

#shared config
parser.add_argument('--labeledtrain_batchsize', default=50, type=int)
parser.add_argument('--unlabeledtrain_batchsize', default=50, type=int)
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")

#PL config
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--lr_warmup_epochs', default=0, type=float,
                    help='warmup epoch for learning rate schedule') #following MixMatch and FixMatch repo

parser.add_argument('--lr_schedule_type', default='CosineLR', choices=['CosineLR', 'FixedLR'], type=str) 
parser.add_argument('--lr_cycle_epochs', default=10000, type=int) #following MixMatch and FixMatch repo


parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--optimizer_type', default='SGD', choices=['SGD', 'Adam'], type=str) 

parser.add_argument('--threshold', default=0.95, type=float, help='confidence threshold')


parser.add_argument('--lambda_u_max', default=1, type=float, help='coefficient of unlabeled loss')

parser.add_argument('--unlabeledloss_warmup_schedule_type', default='NoWarmup', choices=['NoWarmup', 'Linear', 'Sigmoid', ], type=str) 

parser.add_argument('--unlabeledloss_warmup_pos', default=0.4, type=float, help='position at which unlabeled loss warmup ends') #following MixMatch and FixMatch repo




#default hypers not to search for now
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')

parser.add_argument('--use_ema', action='store_true', default=True,
                    help='use EMA model')

parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')

parser.add_argument('--num_classes', default=8, type=int)


#pretrained weights for resnet18
parser.add_argument('--use_pretrained', default='False', type=str)

#early stopping
parser.add_argument('--patience', default=20, type=int, help='Earlystop patience')


def sample_loguniform(low=0, high=1, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    
    return coefficient*np.power(base, power_value)

def sample_uniform(low=0.0, high=1.0, size=1, decimal=1):
    
    return round(np.random.uniform(low=low, high=high, size=size)[0],1)


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise NameError('Bad string')
    

#checked
def save_checkpoint(state, is_best, checkpoint, filename='last_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        
#checked
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
#learning rate schedule   
def get_cosine_schedule_with_warmup(optimizer,
                                    lr_warmup_epochs,
                                    lr_cycle_epochs, #total train epochs
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_epoch):
        if current_epoch < lr_warmup_epochs:
            return float(current_epoch) / float(max(1, lr_warmup_epochs))
        no_progress = float(current_epoch - lr_warmup_epochs) / \
            float(max(1, float(lr_cycle_epochs) - lr_warmup_epochs))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    

def get_fixed_lr(optimizer,
                lr_warmup_epochs,
                lr_cycle_epochs, #total train epochs
                num_cycles=7./16.,
                last_epoch=-1):
    def _lr_lambda(current_epoch):
        
        return 1.0

    return LambdaLR(optimizer, _lr_lambda, last_epoch)    



def create_model(args):
    if args.arch == 'resnet18':
        
        model = models.resnet18(pretrained=args.use_pretrained)
        #https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        model.fc = torch.nn.Linear(512, args.num_classes)
        
    else:
        raise NameError('Note implemented yet')
    

    
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model




def main(args):
    
    #define transform for each part of the dataset
    #SemiSelfEvaluation/check_dataset/MedMNIST/TissueMNIST_original.ipynb
    TissueMNIST_mean = (0.1020, 0.1020, 0.1020)
    TissueMNIST_std = (0.0799, 0.0799, 0.0799)
    
   
    transform_labeledtrain = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                             padding=int(28*0.125),
                             padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=TissueMNIST_mean, std=TissueMNIST_std)
    ])
    
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TissueMNIST_mean, std=TissueMNIST_std)
    ])
    
    
    class TransformTwice:
        def __init__(self, transform_fn):
            self.transform_fn = transform_fn
        
        def __call__(self, x):
            out1 = self.transform_fn(x)
            out2 = self.transform_fn(x)
        
            return out1, out2
    

    l_train_dataset = dataset(args.l_train_dataset_path, transform_fn=transform_labeledtrain)
    u_train_dataset = dataset(args.u_train_dataset_path, transform_fn=TransformTwice(transform_labeledtrain))
    val_dataset = dataset(args.val_dataset_path, transform_fn=transform_eval)
    test_dataset = dataset(args.test_dataset_path, transform_fn=transform_eval)
    
    #########################################for loop from here##################################  
    global_best_hypercombo_ema = dict() 
    global_best_hypercombo_raw = dict()
    
    global_best_val_raw_acc_list_parallel = []
    global_best_test_raw_acc_at_val_list_parallel = []

    global_best_val_ema_acc_list_parallel = []
    global_best_test_ema_acc_at_val_list_parallel = []
    
    
    global_best_val_raw_acc = 0
    global_best_test_raw_acc_at_val = 0
#     global_best_train_raw_acc_at_val = 0

    global_best_val_ema_acc = 0
    global_best_test_ema_acc_at_val = 0
#     global_best_train_ema_acc_at_val = 0

    global_best_val_raw_acc_list = []
    global_best_test_raw_acc_at_val_list = []

    global_best_val_ema_acc_list = []
    global_best_test_ema_acc_at_val_list = []

    elapsed_time_for_each_global_record_list = []

    #create global writer
    global_writer = SummaryWriter(args.train_dir)

    print('!!!!!!!!!!!!!!!!!!!Initial Global Stats!!!!!!!!!!!!!!')
    global_best_val_raw_acc_list.append(global_best_val_raw_acc)
    global_best_test_raw_acc_at_val_list.append(global_best_test_raw_acc_at_val)

    global_best_val_ema_acc_list.append(global_best_val_ema_acc)
    global_best_test_ema_acc_at_val_list.append(global_best_test_ema_acc_at_val)
    elapsed_time_for_each_global_record_list.append(0) #as saintycheck that each global record is around 10h.

    global_writer.add_scalar('global_val/1.global_best_val_raw_acc', global_best_val_raw_acc, 0)
    global_writer.add_scalar('global_val/2.global_best_val_ema_acc', global_best_val_ema_acc, 0)

    global_writer.add_scalar('global_test/1.global_best_test_raw_acc_at_val', global_best_test_raw_acc_at_val, 0)
    global_writer.add_scalar('global_test/2.global_best_test_ema_acc_at_val', global_best_test_ema_acc_at_val, 0)

    #start timing
    hypercombo_iteratethrough_list = [] #newly added
    hypercombo_iteratethrough_list.append(dict())

    current_best_hypercombo_ema_at_each_evaluation_point_list = []
    current_best_hypercombo_ema_at_each_evaluation_point_list.append(dict())
    
    current_best_hypercombo_raw_at_each_evaluation_point_list = []
    current_best_hypercombo_raw_at_each_evaluation_point_list.append(dict())
    
    FiveHourCount_start_time = time.time()
    record_count=0
    total_used_time = 0
    
    #Lr: loguniform base 10, [3*10^(-5) to 3*10^(-2)]
    #Wd: loguniform base 10, [4*10^(-6) to 4*10^(-3)]
    #Lambda_u_max: loguniform base 10, [1*10^(-1) to 1*10^(1)]
    
    while total_used_time <= 100: #run for 100 hour
        lr = sample_loguniform(low=-5, high=-2, size=1, coefficient=3, base=10)
        wd = sample_loguniform(low=-6, high=-3, size=1, coefficient=4, base=10)
        lambda_u_max = sample_loguniform(low=-1, high=1, size=1, coefficient=1, base=10)
        
        print('!!!!!!!!!!This run using lr: {}, wd: {}, lambda_u_max: {}!!!!!!!!!!'.format(lr, wd, lambda_u_max))
        
        hypercombo_iteratethrough_list.append({'lr':lr, 'wd':wd, 'lambda_u_max':lambda_u_max})#newly added
        save_pickle(os.path.join(args.train_dir, 'global_stats'), 'hypercombo_iteratethrough_list.pkl', hypercombo_iteratethrough_list)
        
        this_hypercombo_best_val_raw_acc_list_parallel = []
        this_hypercombo_best_test_raw_acc_at_val_list_parallel = []

        this_hypercombo_best_val_ema_acc_list_parallel = []
        this_hypercombo_best_test_ema_acc_at_val_list_parallel = []
        
        
        ssl_obj = PL(args.threshold, args.num_classes)

        args.lr = lr
        args.wd = wd
        args.lambda_u_max = lambda_u_max

        experiment_name = "Optimizer-{}/LrSchedule-{}_LrCycleEpochs-{}_LrWarmupEpochs-{}/UnlabeledlossWarmupSchedule-{}_UnlabeledlossWarmupPos-{}/LambdaUMax-{}_lr-{}_wd-{}_threshold-{}_em-{}".format(args.optimizer_type, args.lr_schedule_type, args.lr_cycle_epochs, args.lr_warmup_epochs, args.unlabeledloss_warmup_schedule_type, args.unlabeledloss_warmup_pos, args.lambda_u_max, args.lr, args.wd, args.threshold, args.em)

        args.experiment_dir = os.path.join(args.train_dir, 'hypercombos', experiment_name)

        #brief summary:
        brief_summary = {}
        brief_summary['dataset_name'] = args.dataset_name
        brief_summary['algorithm'] = 'PL_RE5'
        brief_summary['hyperparameters'] = {
            'optimizer': args.optimizer_type,
            'lr_schedule_type': args.lr_schedule_type,
            'lr_cycle_epochs': args.lr_cycle_epochs,
            'unlabeledloss_warmup_schedule_type':args.unlabeledloss_warmup_schedule_type,
            'unlabeledloss_warmup_pos': args.unlabeledloss_warmup_pos,
            'lambda_u_max': args.lambda_u_max,
            'lr': args.lr,
            'wd': args.wd,
            'threshold': args.threshold,
        }


        if args.resume != 'None':
            args.resume_checkpoint_fullpath = os.path.join(args.experiment_dir, args.resume)
            print('args.resume_checkpoint_fullpath: {}'.format(args.resume_checkpoint_fullpath))
        else:
            args.resume_checkpoint_fullpath = None

        os.makedirs(args.experiment_dir, exist_ok=True)
        args.writer = SummaryWriter(args.experiment_dir)

        print('Created dataset')
        print("labeled data : {}, unlabeled data : {}, training data : {}".format(
        len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
        print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
    #     brief_summary["number_of_data"] = {
    #     "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
    #     "validation":len(val_dataset), "test":len(test_dataset)
    # }


        l_loader = DataLoader(l_train_dataset, args.labeledtrain_batchsize, shuffle=True, drop_last=True)
        u_loader = DataLoader(u_train_dataset, args.unlabeledtrain_batchsize, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, 128, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, 128, shuffle=False, drop_last=False)


        #create model
        model = create_model(args)
        model.to(args.device)

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wd},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        #optimizer_type choice
        if args.optimizer_type == 'SGD':
            optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                                  momentum=0.9, nesterov=args.nesterov)

        elif args.optimizer_type == 'Adam':
            optimizer = optim.Adam(grouped_parameters, lr=args.lr)

        else:
            raise NameError('Not supported optimizer setting')



        #lr_schedule_type choice
        if args.lr_schedule_type == 'CosineLR':
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)

        elif args.lr_schedule_type == 'FixedLR':
            scheduler = get_fixed_lr(optimizer, args.lr_warmup_epochs, args.lr_cycle_epochs)

        else:
            raise NameError('Not supported lr scheduler setting')


        #instantiate the ema model object
        ema_model = ModelEMA(args, model, args.ema_decay)

        args.start_epoch = 0

        best_val_ema_acc = 0
        best_test_ema_acc_at_val = 0
#                 best_train_ema_acc_at_val = 0

        best_val_raw_acc = 0
        best_test_raw_acc_at_val = 0
#                 best_train_raw_acc_at_val = 0

        this_hypercombo_best_val_raw_acc_list_parallel.append(best_val_raw_acc)
        this_hypercombo_best_test_raw_acc_at_val_list_parallel.append(best_test_raw_acc_at_val)

        this_hypercombo_best_val_ema_acc_list_parallel.append(best_val_ema_acc)
        this_hypercombo_best_test_ema_acc_at_val_list_parallel.append(best_test_ema_acc_at_val)

        current_count=0 #for early stopping, when continue training    
        if args.resume_checkpoint_fullpath is not None:
            try:
                os.path.isfile(args.resume_checkpoint_fullpath)
                logger.info("==> Resuming from checkpoint..")
                checkpoint = torch.load(args.resume_checkpoint_fullpath)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

                best_val_ema_acc = checkpoint['best_val_ema_acc']
                best_test_ema_acc_at_val = checkpoint['best_test_ema_acc_at_val']
#                         best_train_ema_acc_at_val = checkpoint['best_train_ema_acc_at_val']

                best_val_raw_acc = checkpoint['best_val_raw_acc']
                best_test_raw_acc_at_val = checkpoint['best_test_raw_acc_at_val']
#                         best_train_raw_acc_at_val = checkpoint['best_train_raw_acc_at_val']

                current_count = checkpoint['current_count']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print('!!!!Does not have checkpoint yet!!!!')


        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset_name}")
        logger.info(f"  Num Epochs = {args.train_epoch}")
        logger.info(f"  Batch size per GPU (labeled+unlabeled) = {args.labeledtrain_batchsize + args.unlabeledtrain_batchsize}")
        logger.info(f"  Total optimization steps = {args.train_iterations}")


        train_loss_dict = dict()
        train_loss_dict['train_total_loss'] = []
        train_loss_dict['labeled_loss'] = []
        train_loss_dict['unlabeled_loss_unscaled'] = []
        train_loss_dict['unlabeled_loss_scaled'] = []

        is_best = False

        early_stopping = EarlyStopping(patience=args.patience, initial_count=current_count)


        for epoch in range(args.start_epoch, args.train_epoch):
            val_predictions_save_dict = dict()
            test_predictions_save_dict = dict()
    #         train_predictions_save_dict = dict()

            #train
            train_total_loss_list, train_labeled_loss_list, train_unlabeled_loss_unscaled_list, train_unlabeled_loss_scaled_list = train_one_epoch(args, ssl_obj, l_loader, u_loader, model, ema_model, optimizer, scheduler, epoch)

            train_loss_dict['train_total_loss'].extend(train_total_loss_list)
            train_loss_dict['labeled_loss'].extend(train_labeled_loss_list)
            train_loss_dict['unlabeled_loss_unscaled'].extend(train_unlabeled_loss_unscaled_list)
            train_loss_dict['unlabeled_loss_scaled'].extend(train_unlabeled_loss_scaled_list)

            save_pickle(os.path.join(args.experiment_dir, 'losses'), 'losses_dict.pkl', train_loss_dict)

            if epoch % args.eval_every_Xepoch == 0:
                #val
                val_loss, val_raw_acc, val_ema_acc, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, val_loader, model, ema_model.ema, epoch, evaluation_criterion='plain_accuracy')
                val_predictions_save_dict['raw_acc'] = val_raw_acc
                val_predictions_save_dict['ema_acc'] = val_ema_acc
#                 val_predictions_save_dict['true_labels'] = val_true_labels
#                 val_predictions_save_dict['raw_predictions'] = val_raw_predictions
#                 val_predictions_save_dict['ema_predictions'] = val_ema_predictions

                save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'val_epoch_{}_predictions.pkl'.format(str(epoch)), val_predictions_save_dict)

                #test
                test_loss, test_raw_acc, test_ema_acc, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, test_loader, model, ema_model.ema, epoch, evaluation_criterion='plain_accuracy')

                test_predictions_save_dict['raw_acc'] = test_raw_acc
                test_predictions_save_dict['ema_acc'] = test_ema_acc
#                 test_predictions_save_dict['true_labels'] = test_true_labels
#                 test_predictions_save_dict['raw_predictions'] = test_raw_predictions
#                 test_predictions_save_dict['ema_predictions'] = test_ema_predictions

                save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'test_epoch_{}_predictions.pkl'.format(str(epoch)), test_predictions_save_dict)

                #train
    #             train_loss, train_raw_acc, train_ema_acc, train_true_labels, train_raw_predictions, train_ema_predictions = eval_model(args, l_loader, model, ema_model.ema, epoch, evaluation_criterion='plain_accuracy')

    #             train_predictions_save_dict['raw_acc'] = train_raw_acc
    #             train_predictions_save_dict['ema_acc'] = train_ema_acc
    #             train_predictions_save_dict['true_labels'] = train_true_labels
    #             train_predictions_save_dict['raw_predictions'] = train_raw_predictions
    #             train_predictions_save_dict['ema_predictions'] = train_ema_predictions

                if val_raw_acc > best_val_raw_acc:
                    
                    best_val_raw_acc = val_raw_acc
                    best_test_raw_acc_at_val = test_raw_acc
    #                 best_train_raw_acc_at_val = train_raw_acc

                    save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'val_predictions.pkl', val_predictions_save_dict)

                    save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'test_predictions.pkl', test_predictions_save_dict)

    #                 save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'train_predictions.pkl', train_predictions_save_dict)

                if best_val_raw_acc > global_best_val_raw_acc:
                    global_best_hypercombo_raw['lr']=lr
                    global_best_hypercombo_raw['wd']=wd
                    global_best_hypercombo_raw['lambda_u_max']=lambda_u_max
                    
                    global_best_val_raw_acc = best_val_raw_acc
                    global_best_test_raw_acc_at_val = best_test_raw_acc_at_val


                if val_ema_acc > best_val_ema_acc:
                    is_best=True

                    best_val_ema_acc = val_ema_acc
                    best_test_ema_acc_at_val = test_ema_acc
    #                 best_train_ema_acc_at_val = train_ema_acc

                    save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'val_predictions.pkl', val_predictions_save_dict)

                    save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'test_predictions.pkl', test_predictions_save_dict)

    #                 save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'train_predictions.pkl', train_predictions_save_dict)
    
    
                this_hypercombo_best_val_raw_acc_list_parallel.append(best_val_raw_acc)
                this_hypercombo_best_test_raw_acc_at_val_list_parallel.append(best_test_raw_acc_at_val)

                this_hypercombo_best_val_ema_acc_list_parallel.append(best_val_ema_acc)
                this_hypercombo_best_test_ema_acc_at_val_list_parallel.append(best_test_ema_acc_at_val)
                
                

                if best_val_ema_acc > global_best_val_ema_acc:
                    global_best_hypercombo_ema['lr']=lr
                    global_best_hypercombo_ema['wd']=wd
                    global_best_hypercombo_ema['lambda_u_max']=lambda_u_max
                    
                    global_best_val_ema_acc = best_val_ema_acc
                    global_best_test_ema_acc_at_val = best_test_ema_acc_at_val


                elapsed_time = round((time.time() - FiveHourCount_start_time)/ 3600, 2) #in hour
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!elapsed_time: {}'.format(elapsed_time))

                if elapsed_time >=5:
                    record_count += 1
                    total_used_time+=5
                    current_best_hypercombo_ema_at_each_evaluation_point_list.append(global_best_hypercombo_ema.copy())
                    current_best_hypercombo_raw_at_each_evaluation_point_list.append(global_best_hypercombo_raw.copy())
                    
                    global_best_val_raw_acc_list.append(global_best_val_raw_acc)
                    global_best_test_raw_acc_at_val_list.append(global_best_test_raw_acc_at_val)

                    global_best_val_ema_acc_list.append(global_best_val_ema_acc)
                    global_best_test_ema_acc_at_val_list.append(global_best_test_ema_acc_at_val)

                    elapsed_time_for_each_global_record_list.append(elapsed_time)

                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_val_raw_acc_list.pkl', global_best_val_raw_acc_list)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_test_raw_acc_at_val_list.pkl', global_best_test_raw_acc_at_val_list)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_val_ema_acc_list.pkl', global_best_val_ema_acc_list)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_test_ema_acc_at_val_list.pkl', global_best_test_ema_acc_at_val_list)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'elapsed_time_for_each_global_record_list.pkl', elapsed_time_for_each_global_record_list)
                    
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_hypercombo_ema_dict.pkl', global_best_hypercombo_ema)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_hypercombo_raw_dict.pkl', global_best_hypercombo_raw)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'current_best_hypercombo_ema_at_each_evaluation_point_list.pkl', current_best_hypercombo_ema_at_each_evaluation_point_list)
                    save_pickle(os.path.join(args.train_dir, 'global_stats'), 'current_best_hypercombo_raw_at_each_evaluation_point_list.pkl', current_best_hypercombo_raw_at_each_evaluation_point_list)

                    global_writer.add_scalar('global_val/1.global_best_val_raw_acc', global_best_val_raw_acc, record_count)
                    global_writer.add_scalar('global_val/2.global_best_val_ema_acc', global_best_val_ema_acc, record_count)

                    global_writer.add_scalar('global_test/1.global_best_test_raw_acc_at_val', global_best_test_raw_acc_at_val, record_count)
                    global_writer.add_scalar('global_test/2.global_best_test_ema_acc_at_val', global_best_test_ema_acc_at_val, record_count)


                    #reinitialize 
                    FiveHourCount_start_time = time.time()

                logger.info('RAW Best , validation/test %.2f %.2f' % (best_val_raw_acc, best_test_raw_acc_at_val))
                logger.info('EMA Best, validation/test %.2f %.2f' % (best_val_ema_acc, best_test_ema_acc_at_val))

    #                             args.writer.add_scalar('train/1.train_raw_acc', train_raw_acc, epoch)
    #                             args.writer.add_scalar('train/2.train_ema_acc', train_ema_acc, epoch)
                args.writer.add_scalar('train/3.total_loss', np.mean(train_total_loss_list), epoch)
                args.writer.add_scalar('train/4.labeled_loss', np.mean(train_labeled_loss_list), epoch)
                args.writer.add_scalar('train/5.unlabeled_loss_unscaled', np.mean(train_unlabeled_loss_unscaled_list), epoch)
                args.writer.add_scalar('train/6.unlabele_loss_scaled', np.mean(train_unlabeled_loss_scaled_list), epoch)

                args.writer.add_scalar('val/1.val_raw_acc', val_raw_acc, epoch)
                args.writer.add_scalar('val/2.val_ema_acc', val_ema_acc, epoch)
                args.writer.add_scalar('val/3.val_loss', val_loss, epoch)
                args.writer.add_scalar('test/1.test_raw_acc', test_raw_acc, epoch)
                args.writer.add_scalar('test/2.test_ema_acc', test_ema_acc, epoch)
                args.writer.add_scalar('test/3.test_loss', test_loss, epoch)

                brief_summary["number_of_data"] = {
            "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
            "validation":len(val_dataset), "test":len(test_dataset)
        }
                brief_summary['best_test_ema_acc_at_val'] = best_test_ema_acc_at_val 
                brief_summary['best_test_raw_acc_at_val'] = best_test_raw_acc_at_val
    #                             brief_summary['best_train_ema_acc_at_val'] = best_train_ema_acc_at_val 
    #                             brief_summary['best_train_raw_acc_at_val'] = best_train_raw_acc_at_val
                brief_summary['best_val_ema_acc'] = best_val_ema_acc
                brief_summary['best_val_raw_acc'] = best_val_raw_acc

                with open(os.path.join(args.experiment_dir + "brief_summary.json"), "w") as f:
                    json.dump(brief_summary, f)

                #early stopping
                current_count = early_stopping(val_raw_acc)

                save_checkpoint(
                    {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.ema.state_dict(),
                    'best_val_ema_acc': best_val_ema_acc,
                    'best_val_raw_acc': best_val_raw_acc,
                    'best_test_ema_acc_at_val': best_test_ema_acc_at_val,
                    'best_test_raw_acc_at_val': best_test_raw_acc_at_val,
    #                 'best_train_ema_acc_at_val': best_train_ema_acc_at_val,
    #                 'best_train_raw_acc_at_val': best_train_raw_acc_at_val,
                    'current_count':current_count,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, is_best, args.experiment_dir)

                #return is_best to False
                is_best = False

                #early stopping
                if early_stopping.early_stop:
                    break


        brief_summary["number_of_data"] = {
        "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
        "validation":len(val_dataset), "test":len(test_dataset)
    }
        brief_summary['best_test_ema_acc_at_val'] = best_test_ema_acc_at_val 
        brief_summary['best_test_raw_acc_at_val'] = best_test_raw_acc_at_val
    #     brief_summary['best_train_ema_acc_at_val'] = best_train_ema_acc_at_val 
    #     brief_summary['best_train_raw_acc_at_val'] = best_train_raw_acc_at_val
        brief_summary['best_val_ema_acc'] = best_val_ema_acc
        brief_summary['best_val_raw_acc'] = best_val_raw_acc

        global_best_val_raw_acc_list_parallel.append(this_hypercombo_best_val_raw_acc_list_parallel)
        global_best_test_raw_acc_at_val_list_parallel.append(this_hypercombo_best_test_raw_acc_at_val_list_parallel)

        global_best_val_ema_acc_list_parallel.append(this_hypercombo_best_val_ema_acc_list_parallel)
        global_best_test_ema_acc_at_val_list_parallel.append(this_hypercombo_best_test_ema_acc_at_val_list_parallel)

        save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_val_raw_acc_list_parallel.pkl', global_best_val_raw_acc_list_parallel)
        save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_test_raw_acc_at_val_list_parallel.pkl', global_best_test_raw_acc_at_val_list_parallel)
        save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_val_ema_acc_list_parallel.pkl', global_best_val_ema_acc_list_parallel)
        save_pickle(os.path.join(args.train_dir, 'global_stats'), 'global_best_test_ema_acc_at_val_list_parallel.pkl', global_best_test_ema_acc_at_val_list_parallel)
        
        
        args.writer.close()

        with open(os.path.join(args.experiment_dir + "brief_summary.json"), "w") as f:
            json.dump(brief_summary, f)



if __name__ == '__main__':
    args = parser.parse_args()
    
    args.use_pretrained = str2bool(args.use_pretrained)
    args.nimg_per_epoch = 165466 #total size of labeled + unlabeled set for TissueMNIST
        
    cuda = torch.cuda.is_available()
    
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
        torch.backends.cudnn.benchmark = True
    else:
        raise ValueError('Not Using GPU')
    #     device = "cpu"
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.info(dict(args._get_kwargs()))
   
    
    args.train_iterations = args.train_epoch*args.nimg_per_epoch//(args.labeledtrain_batchsize+args.unlabeledtrain_batchsize)
    print('designated train iterations: {}'.format(args.train_iterations))
    
    if args.training_seed is not None:
        print('setting training seed{}'.format(args.training_seed), flush=True)
        set_seed(args.training_seed)

    main(args)
    
    
    
    


    
