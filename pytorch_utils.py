import pandas as pd
from skimage import io
import numpy as np
import random
import PIL
import cv2
import math
from PIL import Image, ImageOps
from torchvision import transforms

class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale = 0.8, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.scale, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = transforms.Scale(self.size, interpolation=self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))


def randomTranspose(img, u=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        img = cv2.transpose(img)
    return img

def randomRotate(img, u=0.25, limit=90):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        angle = random.uniform(-limit,limit)  #degree

        height,width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width/2,height/2),angle,1.0)
        img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        #img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    return img

def randomRotate90(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    d = random.randint(0,4) * 90
    img2 = img.rotate(d, resample=Image.NEAREST)
    return img2

""""
def randomRotate90(img, u=0.25):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        angle=random.randint(0,3)*90
        if angle == 90:
            img = cv2.transpose(img)
            img = cv2.flip(img,1)
            #return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(img,-1)
            #return img[::-1,::-1,:]
        elif angle == 270:
            img = cv2.transpose(img)
            img = cv2.flip(img,0)
            #return  img.transpose((1,0, 2))[::-1,:,:]
    return img
"""
#http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
#http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        img = cv2.flip(img,0)  #np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img

def randomFlip(img, u=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        img = cv2.flip(img,random.randint(-1,1))
    return img


def randomShiftScaleRotate(img, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, u=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        height,width,channel = img.shape

        angle = random.uniform(-rotate_limit,rotate_limit)  #degree
        scale = random.uniform(1-scale_limit,1+scale_limit)
        dx    = round(random.uniform(-shift_limit,shift_limit))*width
        dy    = round(random.uniform(-shift_limit,shift_limit))*height

        cc = math.cos(angle/180*math.pi)*(scale)
        ss = math.sin(angle/180*math.pi)*(scale)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])


        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return img

## unconverntional augmnet ################################################################################3
## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

## barrel\pincushion distortion
def randomDistort1(img, distort_limit=0.35, shift_limit=0.25, u=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        height, width, channel = img.shape

        #debug
        # img = img.copy()
        # for x in range(0,width,10):
        #     cv2.line(img,(x,0),(x,height),(1,1,1),1)
        # for y in range(0,height,10):
        #     cv2.line(img,(0,y),(width,y),(1,1,1),1)

        k  = random.uniform(-distort_limit,distort_limit)  *0.00001
        dx = random.uniform(-shift_limit,shift_limit) * width
        dy = random.uniform(-shift_limit,shift_limit) * height

        #map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
        #https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        #https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        x, y = np.mgrid[0:width:1, 0:height:1]
        x = x.astype(np.float32) - width/2 -dx
        y = y.astype(np.float32) - height/2-dy
        theta = np.arctan2(y,x)
        d = (x*x + y*y)**0.5
        r = d*(1+k*d*d)
        map_x = r*np.cos(theta) + width/2 +dx
        map_y = r*np.sin(theta) + height/2+dy

        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return img


#http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
## grid distortion
def randomDistort2(img, num_steps=10, distort_limit=0.2, u=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if random.random() < u:
        height, width, channel = img.shape

        x_step = width//num_steps
        xx = np.zeros(width,np.float32)
        prev = 0
        for x in range(0, width, x_step):
            start = x
            end   = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step*(1+random.uniform(-distort_limit,distort_limit))

            xx[start:end] = np.linspace(prev,cur,end-start)
            prev=cur


        y_step = height//num_steps
        yy = np.zeros(height,np.float32)
        prev = 0
        for y in range(0, height, y_step):
            start = y
            end   = y + y_step
            if end > width:
                end = height
                cur = height
            else:
                cur = prev + y_step*(1+random.uniform(-distort_limit,distort_limit))

            yy[start:end] = np.linspace(prev,cur,end-start)
            prev=cur


        map_x,map_y =  np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return img


## blur sharpen, etc
def randomFilter(img, limit=0.5, u=0.5):


    if random.random() < u:
        height, width, channel = img.shape

        alpha = limit*random.uniform(0, 1)

        ##kernel = np.ones((5,5),np.float32)/25
        kernel = np.ones((3,3),np.float32)/9*0.2

        # type = random.randint(0,1)
        # if type==0:
        #     kernel = np.ones((3,3),np.float32)/9*0.2
        # if type==1:
        #     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])*0.5

        #kernel = alpha *sharp +(1-alpha)*blur
        #kernel = np.random.randn(5, 5)
        #kernel = kernel/np.sum(kernel*kernel)**0.5

        img = alpha*cv2.filter2D(img, -1, kernel) + (1-alpha)*img
        img = np.clip(img,0.,1.)

    return img


##https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
## color augmentation

#brightness, contrast, saturation-------------
#from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

# def to_grayscle(img):
#     blue  = img[:,:,0]
#     green = img[:,:,1]
#     red   = img[:,:,2]
#     grey = 0.299*red + 0.587*green + 0.114*blue
#     return grey


def randomBrightness(img, limit=0.2, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)
        img = alpha*img
        img = np.clip(img, 0., 1.)
    return img


def randomContrast(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha*img  + gray
        img = np.clip(img,0.,1.)
    return img


def randomSaturation(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587,  0.299]]])
        gray = img * coef
        gray = np.sum(gray,axis=2, keepdims=True)
        img  = alpha*img  + (1.0 - alpha)*gray
        img  = np.clip(img,0.,1.)

    return img

def defaults(imgs):
    return imgs


def rotate90s(imgs):
    return imgs.rotate(90)


def rotate180s(imgs):
    return imgs.rotate(180)


def rotate270s(imgs):
    return imgs.rotate(270)


def horizontalFlips(imgs):
    return imgs.transpose(Image.FLIP_TOP_BOTTOM)


def verticalFlips(imgs):
    return imgs.transpose(Image.FLIP_LEFT_RIGHT)