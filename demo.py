# import the necessary packages
import os
import random

import cv2 as cv
import keras
import keras.backend as K
import numpy as np
from keras.preprocessing.image import (load_img, img_to_array)

import utils
from config import img_rows, img_cols, max_token_length, word2index, start_word, stop_word, vocab_size, words, test_a_image_folder
from model import build_model

if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.00-0.9861.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    names = [f for f in os.listdir(test_a_image_folder) if f.endswith('.jpg')]

    samples = random.sample(names, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_a_image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        img = load_img(filename, target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        print('img_array.shape' + str(img_array.shape))
        image_input = np.array(img_array[0])
        print('image_input.shape' + str(image_input.shape))

        text_input = np.zeros((max_token_length,), dtype=np.int32)
        text_input[0] = word2index[start_word]

        sentence = []
        for i in range(max_token_length):
            output = model.predict([image_input, text_input])
            print('output.shape' + str(output.shape))
            next_index = random.choice(range(vocab_size), p=utils.softmax(output))
            if words[next_index] == stop_word:
                break
            text_input[i + 1] = next_index
            sentence.append(words[next_index])
        print(sentence)

        if not os.path.exists('images'):
            os.makedirs('images')
        bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite('images/{}_image.png'.format(i), bgr)

    K.clear_session()
