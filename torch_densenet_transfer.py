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

plt.ion()  # interactive mode

input_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

n_classes = 17
batch_size = 16

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

X = []
y = []

for f, tags in tqdm(train.values[:], miniters=1000):
    # preprocess input image
    img_path = train_path + "{}.jpg".format(f)
    img = Image.open(img_path)
    img = img.convert('RGB')
    x = input_transform(img)
    # x = np.expand_dims(x, axis=0)
    X.append(x)

    # generate one hot vecctor for label

    targets = np.zeros(n_classes)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y.append(targets)

# X = np.array(X, np.float32)
y = np.array(y, np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
train_data = TensorDataset(torch.stack(X_train), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.stack(X_valid), torch.from_numpy(y_valid))
dsets = {"train": train_data, "val": valid_data}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=2)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = 17

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
                running_acc += multi_f_measure(probs.data, labels).data[0]
                running_loss += loss.data[0]

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

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dset_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


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
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

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

for f, tags in tqdm(test.values[:], miniters=1000):
    img_path = test_path + "{}.jpg".format(f)
    img = Image.open(img_path)
    img = img.convert('RGB')
    x = input_transform(img)
    # x = np.expand_dims(x, axis=0)
    X_test.append(x)

X_test = torch.stack(X_test)
y_test = torch.zeros(X_test.size(0), n_classes)
test_data = TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2)


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
orginin.to_csv('/mnt/home/dunan/Learn/Kaggle/planet_amazon/pytorch_transfer_learning.csv', index=False)
