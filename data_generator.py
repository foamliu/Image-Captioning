import json
import os
import numpy as np
from keras.utils import Sequence

from config import batch_size, img_rows, img_cols
from config import train_folder, train_annotations_filename, valid_folder, valid_annotations_filename


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            annotations_path = os.path.join(train_folder, train_annotations_filename)
        else:
            annotations_path = os.path.join(valid_folder, valid_annotations_filename)

        with open(annotations_path, 'r') as f:
            self.samples = json.load(f)

        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, 1), dtype=np.int32)

        for i_batch in range(length):
            sample = self.samples[i]
            image_id = sample['image_id']

            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
