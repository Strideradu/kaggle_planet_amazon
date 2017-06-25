from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import time
import copy
import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.functional import sigmoid
from pytorch_utils import *
from sklearn.metrics import fbeta_score

size = 256
n_classes = 17
batch_size = 64

input_transform = transforms.Compose([
    transforms.Scale(size + 5),
    transforms.RandomCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: randomRotate90(x)),
    transforms.Lambda(lambda x: randomTranspose(x)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def augment(x, u=0.25):
    if random.random() < u:
        if random.random() > 0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    # x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    return x


input_transform_augmentation = transforms.Compose([
    transforms.Scale(size),
    transforms.Lambda(lambda x: augment(x)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Scale(size),
    transforms.ToTensor()])

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

train_path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/train-jpg/"
test_path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/test-jpg/"
train = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/planet_amazon/train_v2.csv")
test = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/planet_amazon/sample_submission_v2.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

label_map = {'agriculture': 0, 'artisinal_mine': 1, 'bare_ground': 2, 'blooming': 3, 'blow_down': 4, 'clear': 5,
             'cloudy': 6, 'conventional_mine': 7, 'cultivation': 8, 'habitation': 9, 'haze': 10, 'partly_cloudy': 11,
             'primary': 12, 'road': 13, 'selective_logging': 14, 'slash_burn': 15, 'water': 16}
inv_label_map = {i: l for l, i in label_map.items()}

X_train = []
y_train = []
X_valid = []
y_valid = []
y_valid_id = []

for f, tags in train.values[:]:

    targets = np.zeros(n_classes)
    for t in tags.split(' '):
        targets[label_map[t]] = 1

    # preprocess input image
    if random.random() < 0.1:
        img_path = train_path + "{}.jpg".format(f)
        img = Image.open(img_path)
        img = img.convert('RGB')
        x = test_transform(img)
        X_valid.append(x)
        y_valid.append(targets)
        y_valid_id.append(f)
    else:
        img_path = train_path + "{}.jpg".format(f)
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img = np.array(img)
        x = input_transform(img)
        # x = np.expand_dims(x, axis=0)
        X_train.append(x)
        y_train.append(targets)

# X = np.array(X, np.float32)
y_train = np.array(y_train, np.float32)
y_valid = np.array(y_valid, np.float32)

train_data = TensorDataset(torch.stack(X_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.stack(X_valid), torch.from_numpy(y_valid))
dsets = {"train": train_data, "val": valid_data}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                           shuffle=False, num_workers=0)
dset_loaders = {"train": train_loader, "val": valid_loader}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = n_classes

use_gpu = torch.cuda.is_available()


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def multi_criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, labels)
    return loss


def multi_f_measure(probs, labels, threshold=0.235, beta=2):
    SMALL = 1e-6  # 0  #1e-12
    batch_size = probs.size()[0]

    # weather
    l = labels
    p = Variable((probs > threshold).float())

    num_pos = torch.sum(p, 1)
    num_pos_hat = torch.sum(l, 1)
    tp = torch.sum(torch.mul(l, p), 1)
    precise = tp / (num_pos + SMALL)
    recall = tp / (num_pos_hat + SMALL)

    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + SMALL)
    f = fs.sum() / batch_size
    return f


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits = model(inputs)
                probs = sigmoid(logits)
                loss = multi_criterion(logits, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_acc += batch_size * multi_f_measure(probs.data, labels).data[0]
                running_loss += batch_size * loss.data[0]

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_acc / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


######################################################################
# Learning rate scheduler
# ^^^^^^^^^^^^^^^^^^^^^^^
# Let's create our learning rate scheduler. We will exponentially
# decrease the learning rate once every few epochs.

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.85 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

    return optimizer


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, n_classes)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
ignored_params = list(map(id, model_ft.fc.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model_ft.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params},
    {'params': model_ft.fc.parameters(), 'lr': 0.01}
], lr=0.001, momentum=0.9)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

# visualize_model(model_ft)

######################################################################


X_test = []

for f, tags in test.values[:]:
    img_path = test_path + "{}.jpg".format(f)
    img = Image.open(img_path)
    img = img.convert('RGB')
    x = test_transform(img)
    # x = np.expand_dims(x, axis=0)
    X_test.append(x)

X_test = torch.stack(X_test)
y_test = torch.zeros(X_test.size(0), n_classes)
test_data = TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)


def predict(net, test_loader):
    test_dataset = test_loader.dataset
    num_classes = test_dataset.target_tensor.size(1)
    predictions = np.zeros((test_dataset.target_tensor.size(0), num_classes), np.float32)

    test_num = 0
    for iter, (images, indices) in enumerate(test_loader, 0):
        # forward
        logits = net(Variable(images.cuda(), volatile=True))
        probs = sigmoid(logits)

        batch_size = len(images)
        test_num += batch_size
        start = test_num - batch_size
        end = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1, num_classes)

    assert (test_dataset.target_tensor.size(0) == test_num)

    return predictions


model_ft.cuda().eval()
predictions = predict(model_ft, test_loader)
scores = []
for y_pred_row in predictions:

    full_result = []
    for i, value in enumerate(y_pred_row):
        full_result.append(str(i))
        full_result.append(str(value))

    scores.append(" ".join(full_result))

orginin = pd.DataFrame()
orginin['image_name'] = test.image_name.values[:]
orginin['tags'] = scores
orginin.to_csv(
    '/mnt/home/dunan/Learn/Kaggle/planet_amazon/pytorch_resnet50_transfer_learning_scale_augmentation.csv',
    index=False)


######################################################################

# determine best F2 threshold using validation dataset

def get_optimal_threshhold(true_label, prediction, iterations=100):
    best_threshhold = [0.2] * 17
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2] * 17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value

    return best_threshhold


def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')


model_ft.cuda().eval()
valid_predictions = predict(model_ft, dset_loaders["val"])
valid_label = y_valid
f2_threshold = get_optimal_threshhold(valid_label, valid_predictions)
print(f2_threshold)

scores = []
for y_pred_row in valid_predictions:

    full_result = []
    for i, value in enumerate(y_pred_row):
        full_result.append(str(i))
        full_result.append(str(value))

    scores.append(" ".join(full_result))

valid_df = pd.DataFrame()
valid_df['image_name'] = y_valid_id
valid_df['tags'] = scores
valid_df.to_csv(
    '/mnt/home/dunan/Learn/Kaggle/planet_amazon/pytorch_resnet50_transfer_learning_scale_augmentation_valid.csv',
    index=False)
