import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from densenet169 import DenseNet
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Flatten, Input
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import SGD

import scipy
from sklearn.metrics import fbeta_score

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

n_classes = 17

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

base_model = DenseNet(weights_path="/mnt/home/dunan/Learn/Kaggle/planet_amazon/pretrained_weights/densenet169_weights_tf.h5")
x = base_model.get_layer('relu').output

x_newfc = GlobalAveragePooling2D(name='final_pool')(x)
x_newfc = Dense(n_classes, name='fc6')(x_newfc)
x_newfc = Activation('softmax', name='prob')(x_newfc)
model = Model(inputs=base_model.input, outputs=x_newfc)

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

X_train = []
y_train = []

for f, tags in tqdm(train.values[:], miniters=1000):
    # preprocess input image
    img_path = train_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    X_train.append(x)

    # generate one hot vecctor for label

    targets = np.zeros(n_classes)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)



X = np.array(X_train)
y = np.array(y_train, np.uint8)

datagenerator = image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = datagenerator.flow(X, y)
model.fit_generator(generator = train_generator, steps_per_epoch = len(X)/32, epochs = 40)

X_test = []

for f, tags in tqdm(test.values[:], miniters=1000):
    img_path = test_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    X_test.append(x)



X_test = np.array(X_test)
y_pred = model.predict(X_test)

scores = []
for y_pred_row in y_pred:

    full_result = []
    for i, value in enumerate(y_pred_row):
        full_result.append(str(i))
        full_result.append(str(value))

    scores.append(" ".join(full_result))

orginin = pd.DataFrame()
orginin['image_name'] = test.image_name.values
orginin['tags'] = scores
orginin.to_csv('/mnt/home/dunan/Learn/Kaggle/planet_amazon/densenet169_transfer_learning.csv', index=False)