import cv2
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.kgdataset import KgForestDataset, toTensor
from torchvision.transforms import Normalize, Compose, Lambda
import glob
from planet_models.resnet_planet import resnet18_planet, resnet34_planet, resnet50_planet, resnet101_planet, \
    resnet152_planet
from planet_models.fpn import fpn_34, fpn_152, fpn_50
from planet_models.densenet_planet import densenet161, densenet121, densenet169, densenet201
from util import predict, f2_score, pred_csv
from data import kgdataset
from thresholds import thresholds

def cropCenter(img, height, width):

    h,w,c = img.shape
    dx = (h-height)//2
    dy = (w-width )//2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2,x1:x2,:]

    return img

def default(imgs):
    return imgs


def rotate90(imgs):
    for index, img in enumerate(imgs):
        imgs[index] = cv2.transpose(img, (1, 0, 2))
    return imgs


def rotate180(imgs):
    for index, img in enumerate(imgs):
        imgs[index] = cv2.flip(img, -1)
    return imgs


def rotate270(imgs):
    for index, img in enumerate(imgs):
        img = cv2.transpose(img, (1, 0, 2))
        imgs[index] = cv2.flip(img, -1)
    return imgs


def horizontalFlip(imgs):
    for index, img in enumerate(imgs):
        img = cv2.flip(img, 1)
        imgs[index] = img
    return imgs


def verticalFlip(imgs):
    for index, img in enumerate(imgs):
        img = cv2.flip(img, 0)
        imgs[index] = img
    return imgs

def default_224(imgs):
    batch_size = imgs.shape[0]
    new_imgs = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    for index, img in enumerate(imgs):
        new_imgs[index] = cropCenter(img, 224, 224)
    return new_imgs


def rotate90_224(imgs):
    batch_size = imgs.shape[0]
    new_imgs = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    for index, img in enumerate(imgs):
        img = cropCenter(img, 224, 224)
        new_imgs[index] = cv2.transpose(img, (1, 0, 2))
    return new_imgs


def rotate180_224(imgs):
    batch_size = imgs.shape[0]
    new_imgs = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    for index, img in enumerate(imgs):
        img = cropCenter(img, 224, 224)
        new_imgs[index] = cv2.flip(img, -1)
    return new_imgs


def rotate270_224(imgs):
    batch_size = imgs.shape[0]
    new_imgs = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    for index, img in enumerate(imgs):
        img = cropCenter(img, 224, 224)
        img = cv2.transpose(img, (1, 0, 2))
        new_imgs[index] = cv2.flip(img, -1)
    return new_imgs


def horizontalFlip_224(imgs):
    batch_size = imgs.shape[0]
    new_imgs = np.zeros((batch_size,224,224,3),dtype=np.float32)
    for index, img in enumerate(imgs):
        img = cropCenter(img, 224, 224)
        img = cv2.flip(img, 1)
        new_imgs[index] = img
    return new_imgs


def verticalFlip_224(imgs):
    batch_size = imgs.shape[0]
    new_imgs = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    for index, img in enumerate(imgs):
        img = cropCenter(img, 224, 224)
        img = cv2.flip(img, 0)
        new_imgs[index] = img
    return new_imgs


mean = [0.31151703, 0.34061992, 0.29885209]
std = [0.16730586, 0.14391145, 0.13747531]


transforms = [default, rotate90, rotate180, rotate270, verticalFlip, horizontalFlip]

models = [
            # resnet18_planet,
            # resnet34_planet,
            # resnet50_planet,
            # resnet101_planet,
            # resnet152_planet,
            densenet121,
            # densenet161,
            # densenet169,
            # densenet201,
            # fpn_152,
            # fpn_50,
            # fpn_34
        ]


# save probabilities to files for debug
def probs(dataloader):
    """
    returns a numpy array of probabilities (n_transforms, n_models, n_imgs, 17)
    use transforms to find the best threshold
    use models to do ensemble method
    """
    n_transforms = len(transforms)
    n_models = len(models)
    n_imgs = dataloader.dataset.num
    imgs = dataloader.dataset.images.copy()
    probabilities = np.empty((n_transforms, n_models, n_imgs, 17))
    for t_idx, transform in enumerate(transforms):
        t_name = str(transform).split()[1]
        dataloader.dataset.images = transform(imgs)
        for m_idx, model in enumerate(models):
            name = str(model).split()[1]
            net = model().cuda()
            name = 'full_data_{}.pth'.format(name)
            net = nn.DataParallel(net)
            net.load_state_dict(torch.load('/mnt/home/dunan/Learn/Kaggle/planet_amazon/model/{}'.format(name)))
            net.eval()
            # predict
            m_predictions = predict(net, dataloader)

            # save
            np.savetxt(X=m_predictions, fname='/mnt/home/dunan/Learn/Kaggle/planet_amazon/probs/{}_{}.txt'.format(t_name, name))
            probabilities[t_idx, m_idx] = m_predictions
    return probabilities


def find_best_threshold(labels, probabilities):
    threshold = np.zeros(17)
    acc = 0
    # iterate over transformations
    for t_idx in range(len(transforms)):
        # iterate over class labels
        t = np.ones(17) * 0.15
        selected_preds = probabilities[t_idx]
        selected_preds = np.mean(selected_preds, axis=0)
        best_thresh = 0.0
        best_score = 0.0
        for i in range(17):
            for r in range(500):
                r /= 500
                t[i] = r
                preds = (selected_preds > t).astype(int)
                score = f2_score(labels, preds)
                if score > best_score:
                    best_thresh = r
                    best_score = score
            t[i] = best_thresh
        threshold = threshold + t
        acc += best_score
    print('AVG ACC,', acc/len(transforms))
    threshold = threshold / len(transforms)
    return threshold


def get_validation_loader():
    validation = KgForestDataset(
        split='train-40479',
        transform=Compose(
            [
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        height=256,
        width=256
    )
    valid_dataloader = DataLoader(validation, batch_size=16, shuffle=False)
    return valid_dataloader


def get_test_dataloader():
    test_dataset = KgForestDataset(
        split='test-61191',
        transform=Compose(
            [
                Lambda(lambda x: toTensor(x)),
                Normalize(mean=mean, std=std)
            ]
        ),
        label_csv=None
    )

    test_dataloader = DataLoader(test_dataset, batch_size=16)
    return test_dataloader


def do_thresholding(names, models, labels):
    preds = np.empty((len(transforms), len(models), 40479, 17))
    print('filenames', names)
    for t_idx in range(len(transforms)):
        for m_idx in range(len(models)):
            preds[t_idx, m_idx] = np.loadtxt(names[t_idx + m_idx])
    t = find_best_threshold(labels=labels, probabilities=preds)
    return t


def get_files(excludes=None):
    file_names = glob.glob('/mnt/home/dunan/Learn/Kaggle/planet_amazon/probs/*.txt')
    file_names = [f for f in file_names if 'full_data' in f]
    names = []
    for filename in file_names:
        if not any([exclude in filename for exclude in excludes]):
            names.append(filename)
    return names


def predict_test_majority():
    """
    Majority voting method.
    """
    labels = np.empty((len(models), 61191, 17))
    for m_idx, model in enumerate(models):
        name = str(model).split()[1]
        print('predicting model {}'.format(name))
        net = nn.DataParallel(model().cuda())
        net.load_state_dict(torch.load('/mnt/home/dunan/Learn/Kaggle/planet_amazon/model/full_data_{}_10xlr.pth'.format(name)))
        net.eval()
        preds = np.zeros((61191, 17))
        for t in transforms:
            test_dataloader.dataset.images = t(test_dataloader.dataset.images)
            print(t, name)
            pred = predict(net, dataloader=test_dataloader)
            preds = preds + pred
        # get predictions for the single model
        preds = preds/len(transforms)
        np.savetxt('/mnt/home/dunan/Learn/Kaggle/planet_amazon/submission_probs/full_data_{}_single_10xlr_256.txt'.format(name), preds)
        # get labels
        preds = (preds > thresholds[name]).astype(int)
        labels[m_idx] = preds

    # majority voting
    labels = labels.sum(axis=0)
    labels = (labels >= (len(models)//2)).astype(int)
    pred_csv(predictions=labels, name='dense121_single_full_data_10xlr_256')


if __name__ == '__main__':
    # valid_dataloader = get_validation_loader()
    test_dataloader = get_test_dataloader()

    # save results to files
    # probabilities = probs(valid_dataloader)

    # get threshold
    # model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'densenet169']
    # for m in models:
    #     name = str(m).split()[1].strip('_planet')
    #     file_names = get_files([n for n in model_names if n != name])
    #     print('Model {}'.format(name))
    #     t = do_thresholding(file_names, labels=valid_dataloader.dataset.labels, models=[m])
    #     print(t)

    # average testing
    # predict_test_averaging(threshold)

    # majority voting
    predict_test_majority()