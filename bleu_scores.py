# import the necessary packages
import hashlib
import json
import os
import pickle
import sys

import keras.backend as K
import numpy as np
from keras.preprocessing import sequence
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from config import max_token_length, start_word, stop_word, test_a_image_folder, test_a_folder, \
    test_a_annotations_filename
from model import build_model

if __name__ == '__main__':
    channel = 3

    model_weights_path = 'models/model.07-1.5001.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    annotations_path = os.path.join(test_a_folder, test_a_annotations_filename)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

    names = [f for f in encoded_test_a.keys()]

    for image_name in tqdm(names[:5]):
        filename = os.path.join(test_a_image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image_input = np.zeros((1, 2048))
        image_input[0] = encoded_test_a[image_name]
        image_hash = int(int(hashlib.sha256(image_name.split('.')[0].encode('utf-8')).hexdigest(), 16) % sys.maxsize)
        captions = [anno['caption'].split() for anno in annotations['annotations'] if anno['image_id'] == image_hash]

        start_words = [start_word]
        while True:
            text_input = [word2idx[i] for i in start_words]
            text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
            preds = model.predict([image_input, text_input])
            word_pred = idx2word[np.argmax(preds[0])]
            if word_pred == stop_word or len(start_word) >= max_token_length:
                break
            start_words.append(word_pred)

        reference = captions
        candidate = start_words

        print('reference:')
        print(reference)
        print('candidate:')
        print(candidate)

        score = sentence_bleu(reference, candidate)
        print(score)

    K.clear_session()
