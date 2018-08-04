# import the necessary packages
import hashlib
import os
import pickle
import sys
from multiprocessing import Queue

import keras.backend as K
import numpy as np

from tqdm import tqdm

from beam_search import beam_search_predictions
from bleu_worker import InferenceWorker
from config import test_a_image_folder
from utils import get_available_gpus


class Scheduler:
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(InferenceWorker(gpuid, self._queue))

    def start(self, names):

        # put all of files into queue
        for name in names:
            self._queue.put(name)

        # add a None into queue to indicate the end of task
        self._queue.put(None)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")


def run(gpuids):
    # scan all files under img_path
    encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

    names = [f for f in encoded_test_a.keys()]
    names = names[:len(names) // 10]

    # init scheduler
    x = Scheduler(gpuids)

    # start processing and wait for complete
    x.start(names)


if __name__ == "__main__":
    gpuids = range(get_available_gpus())
    print(gpuids)

    run(gpuids)
if __name__ == '__main__':

    total_score = 0
    for image_name in tqdm(names):
        filename = os.path.join(test_a_image_folder, image_name)
        # print('Start processing image: {}'.format(filename))
        image_input = np.zeros((1, 2048))
        image_input[0] = encoded_test_a[image_name]
        image_hash = int(int(hashlib.sha256(image_name.split('.')[0].encode('utf-8')).hexdigest(), 16) % sys.maxsize)
        captions = [anno['caption'].split() for anno in annotations['annotations'] if anno['image_id'] == image_hash]

        reference = captions
        # candidate = start_words
        candidate = beam_search_predictions(model, image_name, word2idx, idx2word, encoded_test_a, beam_index=100)


        # print(score)
        total_score += score

    print('total score: ' + str(total_score))
    print('avg: ' + str(total_score / len(names)))
    K.clear_session()
