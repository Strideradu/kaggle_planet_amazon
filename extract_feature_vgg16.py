import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Input
import pandas as pd
from tqdm import tqdm
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# use vgg 16 model extract feature

base_model = VGG16(weights='imagenet', pooling = max)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

img_path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/train-jpg/train_9.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


features = model.predict(x)
print(features.shape)

# add label
train = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/planet_amazon/train_v2.csv")
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values, miniters=50):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)
    print targets


# save the feature to the tf record file

tfrecords_filename = 'pascal_voc_segmentation.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)