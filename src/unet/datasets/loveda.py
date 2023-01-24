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
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask


def load_image_train(x):
    input_image = tf.image.resize(tf.io.decode_png(
        tf.io.read_file(
            os.path.join(TRAIN_DIR, "images", str(x.numpy())+".png"
    )), channels=3), IMAGE_SIZE)
    input_mask = tf.image.resize(tf.io.decode_png(
        tf.io.read_file(
            os.path.join(TRAIN_DIR, "masks", str(x.numpy())+".png"
    )), channels=1), IMAGE_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_validate(x):
    input_image = tf.image.resize(tf.io.decode_png(
        tf.io.read_file(
            os.path.join(VALIDATE_DIR, "images", str(x.numpy())+".png"
    )), channels=3), IMAGE_SIZE)
    input_mask = tf.image.resize(tf.io.decode_png(
        tf.io.read_file(
            os.path.join(VALIDATE_DIR, "masks", str(x.numpy())+".png"
    )), channels=1), IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_data(train_dir, validate_dir, train_count, validate_count, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    global TRAIN_DIR
    global VALIDATE_DIR
    TRAIN_DIR = train_dir
    VALIDATE_DIR = validate_dir
    train_l = []
    for i in range(train_count):
        train_l.append(i)
    validate_l = []
    for i in range(validate_count):
        validate_l.append(i)
    
    train = tf.data.Dataset.from_tensor_slices(train_l)
    validate = tf.data.Dataset.from_tensor_slices(validate_l)

    train = train.map(load_image_train)
    validate = validate.map(load_image_validate)

    train_dataset = train.cache().shuffle(1000).take(train_count)

    return train_dataset, validate
