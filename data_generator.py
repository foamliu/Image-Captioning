# encoding=utf-8
import json
import os
import pickle

import jieba
import numpy as np
from keras.utils import Sequence

from config import batch_size, max_token_length, start_word, stop_word, unknown_word
from config import train_folder, train_annotations_filename, train_image_folder
from config import valid_folder, valid_annotations_filename, valid_image_folder


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        vocab = pickle.load(open('data/vocab_train.p', 'rb'))
        self.idx2word = sorted(vocab)
        self.word2idx = dict(zip(self.idx2word, range(len(vocab))))

        filename = 'encoded_{}_images.p'.format(usage)
        self.image_encoding = pickle.load(open(filename, 'rb'))

        if usage == 'train':
            annotations_path = os.path.join(train_folder, train_annotations_filename)
            self.image_folder = train_image_folder
        else:
            annotations_path = os.path.join(valid_folder, valid_annotations_filename)
            self.image_folder = valid_image_folder

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        samples = []
        for a in annotations:
            image_id = a['image_id']
            caption = a['caption']
            for c in caption:
                samples.append({'image_id': image_id, 'caption': c})

        self.samples = samples
        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_text_input = np.empty((length, max_token_length), dtype=np.int32)
        batch_image_input = np.empty((length, 2048), dtype=np.float32)
        batch_y = np.empty((length, 1), dtype=np.int32)

        for i_batch in range(length):
            sample = self.samples[i]
            image_id = sample['image_id']
            image_input = np.array(self.image_encoding[image_id])

            caption = sample['caption']
            seg_list = jieba.cut(caption)
            text_input = np.zeros((max_token_length,), dtype=np.int32)
            text_input[0] = self.word2idx[start_word]
            target = np.zeros((max_token_length + 1,), dtype=np.int32)

            for j, word in enumerate(seg_list):
                if word not in self.idx2word:
                    word = unknown_word
                index = self.word2idx[word]
                target[j] = index
                text_input[j + 1] = index
            eos_index = j + 1
            index = self.word2idx[stop_word]
            target[eos_index] = index
            text_input[eos_index + 1] = index

            batch_text_input[i_batch] = text_input
            batch_image_input[i_batch] = image_input
            batch_y[i_batch] = target

            i += 1

        return [batch_image_input, batch_text_input], batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
