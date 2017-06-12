import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Input
import pandas as pd
from tqdm import tqdm
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""

    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


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

label_map = {'agriculture': 0, 'artisinal_mine': 1, 'bare_ground': 2, 'blooming': 3, 'blow_down': 4, 'clear': 5,
             'cloudy': 6, 'conventional_mine': 7, 'cultivation': 8, 'habitation': 9, 'haze': 10, 'partly_cloudy': 11,
             'primary': 12, 'road': 13, 'selective_logging': 14, 'slash_burn': 15, 'water': 16}
inv_label_map = {i: l for l, i in label_map.items()}

# use vgg 16 model extract feature from fc1 layer
base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
input = Input(shape=(224,224,3),name = 'image_input')
x = base_model(input)
# x = base_model.get_layer('avg_pool').output
x = Flatten()(x)
model = Model(inputs=input, outputs=x)

tfrecords_filename = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/extracted_feature/resnet50_flatten_train.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for f, tags in tqdm(train.values[:], miniters=1000):
    # preprocess input image
    img_path = train_path + "{}.jpg".format(f)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # generate feature [2048]
    features = model.predict(x)
    print(features.shape)
    np.squeeze(features)

    # generate one hot vecctor for label

    targets = []
    for t in tags.split(' '):
        targets.append(label_map[t])

    example = tf.train.Example(features=tf.train.Features(feature={
        'video_id': _bytes_feature(f.encode('utf-8')),
        'labels': _int64_feature(targets),
        'rgb': _float_feature(features.tolist()[0])}))

    writer.write(example.SerializeToString())

writer.close()

tfrecords_filename = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/extracted_feature/resnet50_flatten_test.tf_record"
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
    np.squeeze(features)

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
        'rgb': _float_feature(features.tolist()[0])}))

    writer.write(example.SerializeToString())

writer.close()