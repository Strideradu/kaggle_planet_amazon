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

def _bytes_feature_from_string(s):
    return tf.train.Feature(bytes_list=tf.train.BytesList().FromString(s))

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

train_path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/train-jpg/"
test_path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/test-jpg/"
train = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/planet_amazon/train_v2.csv")
test = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/planet_amazon/sample_submission_v2.csv")

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# use vgg 16 model extract feature from fc1 layer
base_model = VGG16(weights='imagenet', pooling = max)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

"""
tfrecords_filename = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/extracted_feature/vgg16_fc1_train.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for f, tags in tqdm(train.values[:], miniters=1000):
    # preprocess input image
    img_path = train_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # generate feature [4096]
    features = model.predict(x)

    # generate one hot vecctor for label
    
    targets = []
    for t in tags.split(' '):
        targets.append(label_map[t])

    # generate file id by replace test to 100000 and file to 200000
    if f.split("_")[0] == "test":
        file_id = int(f.split("_")[-1]) + 1000000
    else:
        file_id= int(f.split("_")[-1]) + 2000000

    example = tf.train.Example(features=tf.train.Features(feature={
        'video_id': _bytes_feature(f.encode('utf-8')),
        'labels': _int64_feature(targets),
        'rgb': _bytes_feature(features.tobytes())}))

    writer.write(example.SerializeToString())

writer.close()

"""

tfrecords_filename = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/extracted_feature/vgg16_fc1_test.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for f, tags in tqdm(test.values[:], miniters=1000):
    # preprocess input image
    img_path = test_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # generate feature [4096]
    features = model.predict(x)

    # generate one hot vecctor for label
    """
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    """
    targets = []

    example = tf.train.Example(features=tf.train.Features(feature={
        'video_id': _bytes_feature(f.encode('utf-8')),
        'labels': _int64_feature(targets),
        'rgb': _bytes_feature(features.tobytes())}))

    writer.write(example.SerializeToString())

writer.close()
