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
    filename = 'encoded_{}_images.p'.format(usage)
    encoding = {}

    if usage == 'train':
        image_folder = train_image_folder
    elif usage == 'valid':
        image_folder = valid_image_folder
    elif usage == 'test_a':
        image_folder = test_a_image_folder
    else: # usage == 'test_b':
        image_folder = test_b_image_folder

    names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    print('starting encoding {} images'.format(usage))
    for i in tqdm(range(len(names))):
        image_name = names[i]
        filename = os.path.join(image_folder, image_name)
        img = load_img(filename, target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        image_input = np.zeros((1, img_rows, img_cols, 3))
        image_input[0] = img_array
        enc = image_model(image_input)
        enc = np.reshape(enc, enc.shape[1])
        encoding[image_name] = enc

    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract('wiki.zh')

    extract(train_folder)
    extract(valid_folder)
    extract(test_a_folder)
    extract(test_b_folder)

    encode_images('train')
    encode_images('valid')
    encode_images('test_a')
    encode_images('test_b')
