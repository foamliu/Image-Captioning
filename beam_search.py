# import the necessary packages
import os
import pickle
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.preprocessing import sequence

from config import max_token_length, test_a_image_folder, img_rows, img_cols
from model import build_model


def beam_search_predictions(image_name, beam_index=3):
    start = [word2idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_token_length:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_token_length, padding='post')
            e = encoding_test[image_name]
            preds = model.predict([np.array([e]), np.array(par_caps)])

            word_preds = np.argsort(preds[0])[-beam_index:]

            # Getting the top <beam_index>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.07-1.5001.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    print(model.summary())

    encoding_test = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

    names = [f for f in encoding_test.keys()]

    samples = random.sample(names, 10)

    for i in range(len(samples)):
        image_name = samples[i]

        image_input = np.zeros((1, 2048))
        image_input[0] = encoding_test[image_name]

        print('Normal Max search:', beam_search_predictions(image_name, beam_index=1))
        print('Beam Search, k=3:', beam_search_predictions(image_name, beam_index=3))
        print('Beam Search, k=5:', beam_search_predictions(image_name, beam_index=5))
        print('Beam Search, k=7:', beam_search_predictions(image_name, beam_index=7))

        filename = os.path.join(test_a_image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        img = cv.imread(filename)
        img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
        if not os.path.exists('images'):
            os.makedirs('images')
        cv.imwrite('images/{}_bs_image.png'.format(i), img)

    K.clear_session()
