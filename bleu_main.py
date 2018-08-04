# import the necessary packages
import hashlib
import json
import os
import pickle
import queue
import sys
from multiprocessing import Process
from multiprocessing import Queue

import keras.backend as K
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from beam_search import beam_search_predictions
from config import best_model, test_a_folder, test_a_annotations_filename
from model import build_model
from utils import get_available_gpus


class InferenceWorker(Process):
    def __init__(self, gpuid, in_queue, out_queue):
        Process.__init__(self, name='ImageProcessor')
        self._gpuid = gpuid
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):

        # set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        # load models
        model = build_model()
        model_weights_path = os.path.join('models', best_model)
        model.load_weights(model_weights_path)

        vocab = pickle.load(open('data/vocab_train.p', 'rb'))
        idx2word = sorted(vocab)
        word2idx = dict(zip(idx2word, range(len(vocab))))

        encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

        annotations_path = os.path.join(test_a_folder, test_a_annotations_filename)

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        while True:
            try:
                image_name = self.in_queue.get(block=False)
            except queue.Empty:
                break
            candidate = beam_search_predictions(model, image_name, word2idx, idx2word, encoded_test_a, beam_index=100)
            image_hash = int(
                int(hashlib.sha256(image_name.split('.')[0].encode('utf-8')).hexdigest(), 16) % sys.maxsize)
            reference = [anno['caption'].split() for anno in annotations['annotations'] if
                         anno['image_id'] == image_hash]
            score = sentence_bleu(reference, candidate)
            self.out_queue.put(score)
            print('woker', self._gpuid, ' image_name ', image_name, " predicted as candidate", candidate)
            print('remaining tasks： ' + str(self.in_queue.qsize()))

        K.clear_session()
        print('InferenceWorker done ', self._gpuid)


class Scheduler:
    def __init__(self, gpuids):
        self.in_queue = Queue()
        self.out_queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(InferenceWorker(gpuid, self.in_queue, self.out_queue))

    def start(self, names):
        # put all of image names into queue
        for name in names:
            self.in_queue.put(name)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")
        return self.out_queue


def run(gpuids):
    # scan all files under img_path
    encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))

    names = [f for f in encoded_test_a.keys()]
    names = names[:len(names) // 10]

    # init scheduler
    x = Scheduler(gpuids)

    # start processing and wait for complete
    return x.start(names)


if __name__ == "__main__":
    gpuids = range(get_available_gpus())
    print(gpuids)

    out_queue = run(gpuids)
    out_list = []
    while out_queue.qsize() > 0:
        out_list.append(out_queue.get())

    print('item number: ' + str(len(out_list)))
    print('total score: ' + str(np.sum(out_list)))
    print('avg: ' + str(np.mean(out_list)))
