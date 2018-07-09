import json
import os
import pickle

import jieba
import tqdm

from config import train_folder, train_annotations_filename
from config import valid_folder, valid_annotations_filename


def build_vocab(usage):
    if usage == 'train':
        annotations_path = os.path.join(train_folder, train_annotations_filename)
    else:
        annotations_path = os.path.join(valid_folder, valid_annotations_filename)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    vocab = set()
    for a in tqdm(annotations):
        caption = a['caption']
        for c in caption:
            seg_list = jieba.cut(c)
            for word in seg_list:
                vocab.add(word)

    filename = 'vocab_{}.p'.format(usage)
    with open(filename, 'wb') as encoded_pickle:
        pickle.dump(vocab, encoded_pickle)


if __name__ == '__main__':
    build_vocab('train')
