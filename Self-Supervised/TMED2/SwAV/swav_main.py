import argparse
import os
import pandas

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset, SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes


import libml.utils as utils
from libml.utils import EarlyStopping
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))


class SwaV(nn.Module):
    def __init__(self, backbone, n_prototypes=512):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=n_prototypes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p


from sklearn.metrics import confusion_matrix as sklearn_cm

def calculate_balanced_accuracy(output, target):
    
    confusion_matrix = sklearn_cm(target, output)
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)
    
    recalls = []
    for i in range(n_class): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    
    return balanced_accuracy * 100

def log_n_uniform(low=-3, high=0, size=1, coefficient=1, base=10):
    power_value = np.random.uniform(low, high, size)[0]
    
    return coefficient*np.power(base, power_value)
    
def uniform(low=0.0, high=1.0, size=1, decimal=1):
    
    return np.random.uniform(low=low, high=high, size=size)[0]
    
    
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def criterion(out_1,out_2,batch_size):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        
        return loss

def train(args, net, data_loader, train_optimizer, scheduler, criterion):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        #print("---------------------------------------------")
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        #print(pos_1, pos_1.shape)
        out_1 = net(pos_1)
        out_2 = net(pos_2)
        print(out_1.shape, out_2.shape)

        loss = criterion([out_1], [out_2])

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        
    print("---------------------------------------------")
    if epoch >= 0:
        scheduler.step()

    return total_loss / total_num



def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
            label_bank.append(target)
            
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        label_bank = torch.cat(label_bank, dim=0).t().contiguous()
        # [N]

        feature_labels = torch.tensor(label_bank, device=feature_bank.device) 
        feature_bank = feature_bank.T.detach().cpu().numpy()
        label_bank = label_bank.numpy()
        clf = LogisticRegression(random_state=0).fit(feature_bank, label_bank)
        
        test_bar = tqdm(test_data_loader)
        label_test_bank = []
        label_test_pred = []
        for data, _, target in test_bar:
            total_num = total_num + len(data)
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data.to(device, non_blocking=True))
            #top_1 = len(np.where(clf.predict(feature.cpu().detach().numpy()) == target.cpu().numpy())[0])
            #total_top1 += top_1
            #print(total_top1)
            label_test_bank = label_test_bank + list(target.cpu().detach().numpy())
            label_test_pred = label_test_pred + list(clf.predict(feature.cpu().detach().numpy()))
            
        balanced_accuracy = calculate_balanced_accuracy(label_test_pred, label_test_bank)
        print(balanced_accuracy)
            

    return balanced_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=10, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-6, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='cifar10', type=str, help='Choose loss function')
    parser.add_argument('--patience', default=10, type=int, help='Earlystop patience')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name


    #configuring an adaptive beta if using annealing method
    
    result = np.zeros(epochs)
    
    # data prepare

    print("Loading data")
    train_data, memory_data, test_data, test_data_2 = utils.get_medical_dataset()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader_2 = DataLoader(test_data_2, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    #train_data, memory_data, test_data = utils.get_medical_dataset()
    print("Data loaded")
    
    # model setup and optimizer config

    c = 9
    print('# Classes: {}'.format(c))

    # training loop
    #os.makedirs('results/{}'.format(dataset_name))

        
    start = time.time()
    epoch_all = 0
    num_hyper_param = 0
    res_10000 = np.zeros((10000, 4))
    hyper_param = np.zeros((100, 2))
    
    while 1:
        lr = log_n_uniform(-4,-2)
        wd = log_n_uniform(-6,-4)
        n_prototypes = int(log_n_uniform(1,3))
        
        hyper_param[num_hyper_param][0] = lr
        hyper_param[num_hyper_param][1] = wd
        num_hyper_param += 1
        np.save("hyper_param", hyper_param)
        
                
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        model = SwaV(backbone)
        model.to(device)
    
        model = SwaV(backbone, n_prototypes)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        print('# Classes: {}'.format(c))
    
        res = np.zeros((epochs, 2))
    
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        best_acc = 0

        train_loss = 0
        val_acc_1 = 0
        test_acc_1 = 0
        epoch = 0
    
            
        results = {'epoch' : [], 'train_loss': [], 'test_acc@1': [], "time": []}
    
        for epoch in range(1, epochs + 1):
            val_acc_1 = test(model, memory_loader, test_loader)
            print(epoch, val_acc_1)
            
            if best_acc < val_acc_1:
                best_acc = val_acc_1
                test_acc_1 = test(model, memory_loader, test_loader_2)

            #test_acc_1 = test(model, memory_loader, test_loader_2)
            train_loss = train(args, model, train_loader, optimizer, scheduler, criterion)
            print(train_loss)
    
            res[epoch-1][0] = float(train_loss)
            res[epoch-1][1] = val_acc_1
            np.save("result_" + 'lr_' + str(lr) + '_wd_' + str(wd) + dataset_name, res)
            
        
            
            error_rate = 1 - val_acc_1
            early_stopping(error_rate, model) # you can replace error_rate with val loss
            if early_stopping.early_stop:
                break
            
            
            res_10000[epoch_all][0] = train_loss
            res_10000[epoch_all][1] = val_acc_1
            res_10000[epoch_all][2] = test_acc_1
            res_10000[epoch_all][3] = (time.time() - start)/60
            np.save("res_10000", res_10000)
            
            epoch_all += 1
            
            
        