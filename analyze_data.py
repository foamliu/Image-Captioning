# encoding=utf-8
import json
import os

import jieba
from tqdm import tqdm

from config import train_folder, train_annotations_filename

if __name__ == '__main__':
    print('Calculating the maximum length among all the captions')
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    with open(annotations_path, 'r') as f:
        samples = json.load(f)

    max_len = 0
    for sample in tqdm(samples):
        caption = sample['caption']
        for c in caption:
            seg_list = jieba.cut(c, cut_all=True)
            if len(seg_list) > max_len:
                max_len = len(seg_list)
    print('max_len: ' + str(max_len))
