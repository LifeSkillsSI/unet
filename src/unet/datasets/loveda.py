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


def load_image_validate(x):
    (validate_dir, number) = x
    input_image = tf.image.resize(tf.io.decode_png(os.path.join(validate_dir, "images", str(number+".png"))), IMAGE_SIZE)
    input_mask = tf.image.resize(tf.io.decode_png(os.path.join(validate_dir, "masks", str(number+".png"))), IMAGE_SIZE)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_data(train_dir, validate_dir, train_count, validate_count, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_l = []
    for i in range(train_count):
        train_l.append((train_dir, i))
    validate_l = []
    for i in range(validate_count):
        validate_l.append((validate_dir, i))
    
    train = tf.data.Dataset.from_tensor_slices(train_l)
    validate = tf.data.Dataset.from_tensor_slices(validate_l)

    train = train.map(load_image_train)
    validate = validate.map(load_image_validate)

    return train, validate
