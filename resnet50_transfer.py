import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input, Dense
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

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# use vgg 16 model extract feature from fc1 layer
base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
x = base_model.output
x = Flatten()(x)
predictions = Dense(17, activation='softmax', name='fc1000')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'], loss='categorical_crossentropy')


X_train = []
y_train = []

for f, tags in tqdm(train.values[:], miniters=1000):
    # preprocess input image
    img_path = train_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
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
    x = np.expand_dims(x, axis=0)
    X_test.append(x)



X_test = np.array(X_test)
y_pred = model.predict(X_test)

orginin = pd.DataFrame()
orginin['image_name'] = test.image_name.values
orginin['tags'] = y_pred
orginin.to_csv('/mnt/home/dunan/Learn/Kaggle/planet_amazon/rsenet50_trnsfer_learning.csv', index=False)