from typing import Tuple, Dict
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo

tfds.disable_progress_bar()

IMAGE_SIZE = (256, 256)
channels = 3
classes = 7

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image_train(x):
    (train_dir, number) = x
    input_image = tf.image.resize(tf.io.decode_png(os.path.join(train_dir, "images", str(number+".png"))), IMAGE_SIZE)
    input_mask = tf.image.resize(tf.io.decode_png(os.path.join(train_dir, "masks", str(number+".png"))), IMAGE_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(x):
    (test_dir, number) = x
    input_image = tf.image.resize(tf.io.decode_png(os.path.join(test_dir, "images", str(number+".png"))), IMAGE_SIZE)
    input_mask = tf.image.resize(tf.io.decode_png(os.path.join(test_dir, "masks", str(number+".png"))), IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_data(train_dir, test_dir, train_count, test_count, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_l = []
    for i in range(train_count):
        train_l.append((train_dir, i))
    test_l = []
    for i in range(test_count):
        test_l.append((test_dir, i))
    
    train = tf.data.Dataset.from_tensor_slices(train_l)
    test = tf.data.Dataset.from_tensor_slices(test_l)

    train = train.map(load_image_train)
    test = test.map(load_image_test)

    return train, test
