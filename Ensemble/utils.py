import settings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os, glob
import cv2
import random
import argparse
import bcolz
import pandas as pd
import random
from PIL import Image
from inception import inception_v3
from vgg import vgg19_bn, vgg16_bn
from sklearn.metrics import fbeta_score

MODEL_DIR = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/model/"

def get_multi_classes(score, classes, threshold, nil=''):
    N = len(classes)
    if not isinstance(threshold,list) : threshold = [threshold]*N
    s=nil
    for n in range(N):
        if score[n]>threshold[n]:
            if s==nil:
                s = classes[n]
            else:
                s = '%s %s'%(s,classes[n])
    return s

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

w_files_training = []

def get_acc_from_w_filename(filename):
    try:
        stracc = filename.split('_')[-2]
        return float(stracc)
    except:
        return 0.

def load_best_weights(model):
    w_files = glob.glob(os.path.join(MODEL_DIR, model.name) + '_*.pth')
    max_acc = 0
    best_file = None
    for w_file in w_files:
        try:
            stracc = w_file.split('_')[-2]
            acc = float(stracc)
            if acc > max_acc:
                best_file = w_file
                max_acc = acc
            w_files_training.append((acc, w_file))
        except:
            continue
    if max_acc > 0:
        print('loading weight: {}'.format(best_file))
        model.load_state_dict(torch.load(best_file))

def save_weights(acc, model, epoch, max_num=2):
    f_name = '{}_{}_{:.5f}_.pth'.format(model.name, epoch, acc)
    w_file_path = os.path.join(MODEL_DIR, f_name)
    if len(w_files_training) < max_num:
        w_files_training.append((acc, w_file_path))
        torch.save(model.state_dict(), w_file_path)
        return
    min = 10.0
    index_min = -1
    for i, item in enumerate(w_files_training):
        val_acc, fp = item
        if min > val_acc:
            index_min = i
            min = val_acc
    #print(min)
    if acc > min:
        torch.save(model.state_dict(), w_file_path)
        try:
            os.remove(w_files_training[index_min][1])
        except:
            print('Failed to delete file: {}'.format(w_files_training[index_min][1]))
        w_files_training[index_min] = (acc, w_file_path)

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def load_weights_file(model, w_file):
    model.load_state_dict(torch.load(w_file))

def create_res50(load_weights=False):
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 17)
    model_ft = model_ft.cuda()

    model_ft.name = 'res50'
    model_ft.batch_size = 32
    return model_ft

def create_res101(load_weights=False):
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 17)
    model_ft = model_ft.cuda()

    model_ft.name = 'res101'
    model_ft.batch_size = 32
    model_ft.max_num = 2
    return model_ft

def create_res152(load_weights=False):
    res152 = models.resnet152(pretrained=True)
    num_ftrs = res152.fc.in_features
    res152.fc = nn.Linear(num_ftrs, 17)
    res152 = res152.cuda()

    res152.name = 'res152'
    res152.max_num = 2
    return res152

def create_dense161(load_weights=False):
    desnet_ft = models.densenet161(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 17)
    desnet_ft = desnet_ft.cuda()

    desnet_ft.name = 'dense161'
    desnet_ft.max_num = 2
    #desnet_ft.batch_size = 32
    return desnet_ft

def create_dense169(load_weights=False):
    desnet_ft = models.densenet169(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 17)
    desnet_ft = desnet_ft.cuda()

    desnet_ft.name = 'dense169'
    #desnet_ft.batch_size = 32
    return desnet_ft

def create_dense121(load_weights=False):
    desnet_ft = models.densenet121(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 17)
    desnet_ft = desnet_ft.cuda()

    desnet_ft.name = 'dense121'
    desnet_ft.batch_size = 32
    return desnet_ft

def create_dense201(load_weights=False):
    desnet_ft = models.densenet201(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 17)
    desnet_ft = desnet_ft.cuda()
 
    desnet_ft.name = 'dense201'
    desnet_ft.max_num = 2
    #desnet_ft.batch_size = 32
    return desnet_ft

def create_vgg19bn(load_weights=False):
    vgg19_bn_ft = vgg19_bn(pretrained=True)
    #vgg19_bn_ft.classifier = nn.Linear(25088, 3)
    vgg19_bn_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 17))

    vgg19_bn_ft = vgg19_bn_ft.cuda()

    vgg19_bn_ft.name = 'vgg19bn'
    vgg19_bn_ft.max_num = 1
    #vgg19_bn_ft.batch_size = 32
    return vgg19_bn_ft

def create_vgg16bn(load_weights=False):
    vgg16_bn_ft = vgg16_bn(pretrained=True)
    #vgg16_bn_ft.classifier = nn.Linear(25088, 3)
    vgg16_bn_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 17))

    vgg16_bn_ft = vgg16_bn_ft.cuda()

    vgg16_bn_ft.name = 'vgg16bn'
    vgg16_bn_ft.max_num = 1
    #vgg16_bn_ft.batch_size = 32
    return vgg16_bn_ft

def create_inceptionv3(load_weights=False):
    incept_ft = inception_v3(pretrained=True)
    num_ftrs = incept_ft.fc.in_features
    incept_ft.fc = nn.Linear(num_ftrs, 17)
    incept_ft.aux_logits=False
    incept_ft = incept_ft.cuda()

    incept_ft.name = 'inceptionv3'
    incept_ft.batch_size = 32
    return incept_ft

def create_vgg19(load_weights=False):
    vgg19_ft = models.vgg19(pretrained=True)
    vgg19_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 17))
    vgg19_ft = vgg19_ft.cuda()

    vgg19_ft.name = 'vgg19'
    vgg19_ft.batch_size = 32
    vgg19_ft.max_num = 1
    return vgg19_ft

def create_vgg16(load_weights=False):
    vgg16_ft = models.vgg16(pretrained=True)
    vgg16_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 17))
    vgg16_ft = vgg16_ft.cuda()

    vgg16_ft.name = 'vgg16'
    vgg16_ft.batch_size = 32
    return vgg16_ft

def create_model(model_name):
    create_func = 'create_' + model_name

    return eval(create_func)()
