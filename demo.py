# import the necessary packages
import os
import random

import cv2 as cv
import keras
import keras.backend as K
import numpy as np
from keras.preprocessing.image import (load_img, img_to_array)

from config import img_rows, img_cols, max_token_length, word2index, start_word
from model import build_model

if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.08-5.7380.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    image_folder = '/data/ai_challenger_caption_test_a_20180103'
    names = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    samples = random.sample(names, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        img = load_img(filename, target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        image_input = np.array(img_array[0])

        text_input = np.zeros((max_token_length,), dtype=np.int32)
        text_input[0] = word2index[start_word]

        for i in range(max_token_length):
            None

        if not os.path.exists('images'):
            os.makedirs('images')
        bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite('images/{}_image.png'.format(i), bgr)

    K.clear_session()
