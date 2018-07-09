import json
import os
import pickle

import jieba
from tqdm import tqdm

from config import start_word, stop_word
from config import train_folder, train_annotations_filename
from config import valid_folder, valid_annotations_filename


def build_train_vocab(usage):
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    print('building {} train vocab')
    vocab = set()
    for a in tqdm(annotations):
        caption = a['caption']
        for c in caption:
            seg_list = jieba.cut(c)
            for word in seg_list:
                vocab.add(word)

    vocab.add(start_word)
    vocab.add(stop_word)

    filename = 'data/vocab_{}.p'.format(usage)
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(vocab, encoded_pickle)


def build_samples(usage):
    if usage == 'train':
        annotations_path = os.path.join(train_folder, train_annotations_filename)
    else:
        annotations_path = os.path.join(valid_folder, valid_annotations_filename)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = list(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    print('building {} samples'.format(usage))
    samples = []
    for a in tqdm(annotations):
        image_id = a['image_id']
        caption = a['caption']
        for c in caption:
            seg_list = jieba.cut(c)
            input = []
            last_word = start_word
            for j, word in enumerate(seg_list):
                input.append(word2idx[last_word])
                samples.append({'image_id': image_id, 'input': input, 'output': word2idx[word]})
                last_word = word
            input.append(word2idx[last_word])
            samples.append({'image_id': image_id, 'input': input, 'output': word2idx[stop_word]})

    filename = 'data/samples_{}.p'.format(usage)
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    if not os.path.isfile('data/vocab_train.p'):
        build_train_vocab('train')

    if not os.path.isfile('data/samples_train.p'):
        build_samples('train')

    if not os.path.isfile('data/samples_valid.p'):
        build_samples('valid')
