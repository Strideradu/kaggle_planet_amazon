import os, cv2, glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import random

# import transforms

DATA_DIR = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/"
TRAIN_DIR = DATA_DIR + '/train-jpg'
TEST_DIR = DATA_DIR + '/test-jpg'

classes = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear',
           'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
           'primary', 'road', 'selective_logging', 'slash_burn', 'water']
label_map = {l: i for i, l in enumerate(classes)}


def pil_load(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class PlanetDataset(data.Dataset):
    def __init__(self, file_list_path, train_data=True, has_label=True, transform=None, split=0.9):
        df_train = pd.read_csv(file_list_path)
        if has_label:
            split_index = (int)(df_train.values.shape[0] * split)
            if train_data:
                split_data = df_train.values[:split_index]
            else:
                split_data = df_train.values[split_index:]

            filenames = [None] * split_data.shape[0]
            labels = [None] * split_data.shape[0]

            for i, line in enumerate(split_data):
                f, tags = line
                filenames[i] = TRAIN_DIR + '/' + f + '.jpg'
                labels[i] = np.zeros(17)
                for t in tags.split(' '):
                    labels[i][label_map[t]] = 1
        else:
            filenames = [None] * df_train.values.shape[0]
            for i, line in enumerate(df_train.values):
                f, tags = line

                filenames[i] = TEST_DIR + '/' + f + '.jpg'

        self.transform = transform
        self.num = len(filenames)
        self.filenames = filenames
        self.classes = classes
        self.train_data = train_data
        self.has_label = has_label

        if has_label:
            self.labels = np.array(labels, dtype=np.float32)
            print(self.labels.shape)

        print(self.num)

    def __getitem__(self, index):
        img = pil_load(self.filenames[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.has_label:
            label = self.labels[index]
            return img, label, self.filenames[index]
        else:
            return img, self.filenames[index]
            # if self.train_data:
            #    label = self.labels[index]
            #    return img, label, self.filenames[index]
            # else:
            #    return img, self.filenames[index]

    def __len__(self):
        # print ('\tcalling Dataset:__len__')
        return self.num


def randomRotate(img):
    d = random.randint(0, 4) * 90
    img2 = img.rotate(d, resample=Image.NEAREST)
    return img2


data_transforms = {
    'train': transforms.Compose([
        # transforms.Scale(320),
        # transforms.RandomSizedCrop(224),
        transforms.Scale(224),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'trainv3': transforms.Compose([
        transforms.Scale(299),
        # transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: randomRotate(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validv3': transforms.Compose([
        transforms.Scale(299),
        # transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testv3': transforms.Compose([
        transforms.Scale(299),
        # transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

'''
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'valid']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'valid']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
dset_classes = dsets['train'].classes
save_array(CLASSES_FILE, dset_classes)
'''


def get_train_loader(model, batch_size=16, shuffle=True):
    if model.name == 'inceptionv3':
        transkey = 'trainv3'
    else:
        transkey = 'train'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    # train_v2.csv
    print('batch size:{}'.format(batch_size))
    dset = PlanetDataset(DATA_DIR + '/train_v2.csv', transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader


def get_val_loader(model, batch_size=16, shuffle=True):
    if model.name == 'inceptionv3':
        transkey = 'validv3'
    else:
        transkey = 'valid'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    print('batch size:{}'.format(batch_size))
    # train_v2.csv
    dset = PlanetDataset(DATA_DIR + '/train_v2.csv', train_data=False, transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader


def get_test_loader(model, batch_size=16, shuffle=False):
    if model.name == 'inceptionv3':
        transkey = 'testv3'
    else:
        transkey = 'test'
    if hasattr(model, 'batch_size'):
        batch_size = model.batch_size
    print('batch size:{}'.format(batch_size))

    dset = PlanetDataset(DATA_DIR + '/sample_submission_v2.csv', has_label=False, transform=data_transforms[transkey])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    dloader.num = dset.num
    return dloader


if __name__ == '__main__':
    loader = get_train_loader()
    print(loader.num)
    for i, data in enumerate(loader):
        img, label, fn = data
        # print(fn)
        # print(label)
        if i > 10:
            break
    loader = get_val_loader()
    print(loader.num)
    for i, data in enumerate(loader):
        img, label, fn = data
        # print(fn)
        # print(label)
        if i > 10:
            break
    loader = get_test_loader()
    print(loader.num)
    for i, data in enumerate(loader):
        img, fn = data
        # print(fn)
        # print(label)
        if i > 10:
            break
