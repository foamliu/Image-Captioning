import os
import json
from tqdm import tqdm

from config import train_folder, train_annotations_filename

if __name__ == '__main__':
    annotations_path = os.path.join(train_folder, train_annotations_filename)

    with open(annotations_path, 'r') as f:
        samples = json.load(f)

    max_len = 0
    for sample in tqdm(samples):
        caption = sample['caption']
        for c in caption:
            if len(c) > max_len:
                max_len = len(c)

    print('max_len: ' + str(max_len))
