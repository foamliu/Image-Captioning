# import the necessary packages
import os
import pickle
import random

import cv2 as cv
import keras
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.image import (load_img, img_to_array)

from config import img_rows, img_cols, max_token_length, start_word, stop_word, test_a_image_folder
from model import build_model

if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.00-1.9058.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    print(model.summary())

    names = [f for f in os.listdir(test_a_image_folder) if f.endswith('.jpg')]

    samples = random.sample(names, 1)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_a_image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        img = load_img(filename, target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        print('img_array.shape: ' + str(img_array.shape))
        img_array = keras.applications.resnet50.preprocess_input(img_array)
        image_input = np.zeros((1, 224, 224, 3))
        image_input[0] = img_array

        start_words = [start_word]
        while True:
            text_input = [word2idx[i] for i in start_words]
            text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
            preds = model.predict([image_input, text_input])
            # print('output.shape: ' + str(output.shape))
            word_pred = idx2word[np.argmax(preds[0])]
            start_words.append(word_pred)
            if word_pred == stop_word or len(start_word) > max_token_length:
                break

        sentence = ' '.join(start_words[1:-1])
        print(sentence)

        if not os.path.exists('images'):
            os.makedirs('images')
        bgr = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
        cv.imwrite('images/{}_image.png'.format(i), bgr)

    K.clear_session()
