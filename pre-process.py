import os
import pickle
import zipfile

import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import (load_img, img_to_array)
from tqdm import tqdm

from config import img_rows, img_cols
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_image_folder, valid_image_folder, test_a_image_folder, test_b_image_folder

image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def encode_images(usage):
    encoding = {}

    if usage == 'train':
        image_folder = train_image_folder
    elif usage == 'valid':
        image_folder = valid_image_folder
    elif usage == 'test_a':
        image_folder = test_a_image_folder
    else:  # usage == 'test_b':
        image_folder = test_b_image_folder

    names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    print('encoding {} images'.format(usage))
    for i in tqdm(range(len(names))):
        image_name = names[i]
        filename = os.path.join(image_folder, image_name)
        img = load_img(filename, target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        image_input = np.zeros((1, img_rows, img_cols, 3))
        image_input[0] = img_array
        enc = image_model.predict(image_input)
        enc = np.reshape(enc, enc.shape[1])
        encoding[image_name] = enc

    filename = 'data/encoded_{}_images.p'.format(usage)
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    if not os.path.isfile('data/wiki.zh.vec'):
        extract('data/wiki.zh')

    if not os.path.isdir(train_image_folder):
        extract(train_folder)

    if not os.path.isdir(valid_image_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_image_folder):
        extract(test_a_folder)

    if not os.path.isdir(test_b_image_folder):
        extract(test_b_folder)

    if not os.path.isfile('encoded_train_images.p'):
        encode_images('train')

    if not os.path.isfile('encoded_valid_images.p'):
        encode_images('valid')

    if not os.path.isfile('encoded_test_a_images.p'):
        encode_images('test_a')

    if not os.path.isfile('encoded_test_b_images.p'):
        encode_images('test_b')
