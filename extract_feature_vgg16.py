import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Flatten, Input
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# use vgg 16 model extract feature

model = VGG16(weights='imagenet', include_top=False, pooling = max)

img_path = "/mnt/home/dunan/Learn/Kaggle/planet_amazon/train-jpg/train_9.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


features = model.predict(x)
print(features.shape)


# save the feature to the tf record file

tfrecords_filename = 'pascal_voc_segmentation.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)