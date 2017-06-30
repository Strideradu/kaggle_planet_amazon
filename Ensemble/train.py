import settings

import torch
import numpy as np
import random

random.seed(1234)
torch.manual_seed(5678)
np.random.seed(2468)
#torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import cv2
import argparse
import bcolz
import pandas as pd
from PIL import Image
from sklearn.metrics import fbeta_score
import torch.nn.functional as F

from utils import save_array, load_array, save_weights, load_best_weights, w_files_training
from utils import create_model
from cscreendataset import get_train_loader, get_val_loader
from utils import optimise_f2_thresholds

data_dir = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/"
MODEL_DIR = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/model/"

RESULT_DIR = data_dir + '/results'
THRESHOLD_FILE = RESULT_DIR + '/best_threshold.dat'
#CLASSES_FILE = RESULT_DIR + '/train_classes.dat'

batch_size = 16
epochs = 20

thr = [0.21, 0.25, 0.09, 0.07, 0.27, 0.21, 0.23, 0.24, 0.22, 0.21, 0.16, 0.07, 0.13, 0.1, 0.26, 0.39, 0.03]

def logits_to_probs(logits, is_force_single_weather=False):

    probs = F.sigmoid(Variable(logits)).data
    if is_force_single_weather:
        weather = logits[:, 0:4]
        maxs, indices = torch.max(weather, 1)
        weather.zero_()
        weather.scatter_(1, indices, 1)
        probs[:, 0:4] = weather
        # print(probs)

    return probs


def f_measure(logits, labels, threshold=0.23, beta=2):

    SMALL = 1e-6  # 0  #1e-12
    batch_size = logits.size()[0]

    # weather
    probs = logits_to_probs(logits)
    l = labels
    #p = (probs > threshold).float()

    p = torch.zeros(len(probs), len(probs[0])).cuda()
    for i in range(17):
        p[:, i] = probs[:, i] > thr[i]

    num_pos = torch.sum(p,  1) + SMALL
    num_pos_hat = torch.sum(l,  1)
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat

    fs = (1 + beta * beta) * precise * recall / \
        (beta * beta * precise + recall + SMALL)
    f = fs.sum() / batch_size
    return f

def criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, labels)
    return loss

def evaluate(net, test_loader):
    
    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, fn) in enumerate(test_loader, 0):
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        # forward
        #print(iter)
        #print(images.size())
        logits = net(images)
        loss   = criterion(logits, labels)

        batch_size = len(images)
        test_acc  += batch_size*f_measure(logits.data, labels.data)
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size
        #del logits
        #del labels

    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc, test_num


def train_model(model, criterion, optimizer, lr_scheduler, max_num=2, init_lr=0.001, num_epochs=100):
    #data_loaders = { 'train' : get_train_loader(), 'valid': get_val_loader()}
    #save_array(CLASSES_FILE, data_loaders['train'].classes)
    train_loader = get_train_loader(model)
    val_loader = get_val_loader(model)

    since = time.time()
    best_model = model
    best_acc = 0.0
    print(model.name)

    print('** start training here! **')
    print(' epoch   iter   rate  |  smooth_loss  train_loss  (acc)  |  valid_loss    (acc)   | min')
    print('---------------------------------------------------------------------------------------')

    for epoch in range(num_epochs):
        epoch_since = time.time()
        start = time.time()
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        #for phase in ['train', 'valid']:
        if True:
            optimizer = lr_scheduler(optimizer, epoch, init_lr=init_lr)
            model.train(True)  # Set model to training mode
            
            for param_group in optimizer.param_groups:
                rate = param_group['lr']

            running_loss = 0.0
            running_corrects = 0
            num = 0
            for it, data in enumerate(train_loader, 0):
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                num += 1
            
                if (it+1) % 100 == 0:
                    smooth_loss = running_loss/num
                    running_loss = 0.0
                    num = 0
                    train_acc = f_measure(outputs.data, labels.data)
                    train_loss = loss.data[0]
                    print('\r%5.1f   %5d    %0.4f   |  %0.4f  %0.4f  %5.4f | ... ' % \
                        (epoch, it + 1, rate, smooth_loss, train_loss, train_acc),\
                        end='',flush=True)

               # train_acc  = f_measure(logits.data, labels.cuda())
               # train_loss = loss.data[0]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        end = time.time()
        epoch_time = (end - start)/60
        if True:
            model.train(False)
            #model.eval()
            test_loss,test_acc,test_num = evaluate(model, val_loader)
            assert(test_num==val_loader.num)
            
            print('\r',end='',flush=True)
            print('%5.1f   %5d    %0.4f   |  %0.4f  %0.4f  %5.4f | %0.4f  %5.4f  |  %3.1f min' % \
                    (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss,test_acc, epoch_time))
            
            save_weights(test_acc, model, epoch, max_num=max_num)
            
        #print('epoch {}: {:.0f}s'.format(epoch, time.time() - epoch_since))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    #print(w_files_training)
    return model


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.6**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        print('existing lr = {}'.format(param_group['lr']))
        param_group['lr'] = lr
    return optimizer


def cyc_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=2):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        lr = lr * 0.8
    if lr < 5e-6:
        lr = 0.0001
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer


def train(model, init_lr=0.0005, num_epochs=epochs):
    # nn.CrossEntropyLoss() nn.MultiLabelMarginLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    #optimizer_ft = optim.Adam(model.parameters(), lr=init_lr)

    model = train_model(model, criterion, optimizer_ft, cyc_lr_scheduler, init_lr=init_lr,
                        num_epochs=num_epochs, max_num=model.max_num)
    return model


def train_net(model_name):
    print('Training {}...'.format(model_name))
    model = create_model(model_name)
    try:
        load_best_weights(model)
    except:
        print('Failed to load weigths')
    if not hasattr(model, 'max_num'):
        model.max_num = 1
    train(model)

    

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs=1, help="train model")
parser.add_argument("--threshold", action='store_true', help="train model")

args = parser.parse_args()
if args.train:
    print('start training model')
    mname = args.train[0]
    train_net(mname)

    print('done')
if args.threshold:
    find_best_threshold()
