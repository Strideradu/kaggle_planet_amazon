import pandas as pd
from skimage import io
import numpy as np
import random
import PIL
import cv2

def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = cv2.transpose(img)
    return img

def randomRotate90(img, u=0.25):
    if random.random() < u:
        angle=random.randint(1,3)*90
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