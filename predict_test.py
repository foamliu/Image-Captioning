# import the necessary packages
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import sys
from multiprocessing import Process

import numpy as np
from tqdm import tqdm

from config import best_model, test_a_folder, test_a_annotations_filename, beam_size

encoded_test_a = pickle.load(open('data/encoded_test_a_images.p', 'rb'))


class InferenceWorker(Process):
    def __init__(self, gpuid, in_queue, out_queue, signal_queue):
        Process.__init__(self, name='ImageProcessor')

        self.gpuid = gpuid
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.signal_queue = signal_queue

    def run(self):
        # set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        print("InferenceWorker init, GPU ID: {}".format(self.gpuid))

        from model import build_model
        from beam_search import beam_search_predictions

        # load models
        model = build_model()
        model_weights_path = os.path.join('models', best_model)
        model.load_weights(model_weights_path)

        vocab = pickle.load(open('data/vocab_train.p', 'rb'))
        idx2word = sorted(vocab)
        word2idx = dict(zip(idx2word, range(len(vocab))))

        while True:
            try:
                try:
                    image_name = self.in_queue.get(block=False)
                    num_done = self.out_queue.qsize()
                except queue.Empty:
                    continue

                candidate = beam_search_predictions(model, image_name, word2idx, idx2word, encoded_test_a,
                                                    beam_index=beam_size)

                self.out_queue.put({'image_name': image_name, 'candidate': candidate})
                self.signal_queue.put(SENTINEL)

                if num_done % 1000 == 0:
                    with open("preds_{}.p".format(num_done), "wb") as file:
                        pickle.dump(self.out_queue, file)

                if self.in_queue.qsize() == 0:
                    break
            except Exception as e:
                print(e)

        import keras.backend as K
        K.clear_session()
        print('InferenceWorker done, GPU ID {}'.format(self.gpuid))


class Scheduler:
    def __init__(self, gpuids, signal_queue):
        self.signal_queue = signal_queue
        self.in_queue = manager.Queue()
        self.out_queue = manager.Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(InferenceWorker(gpuid, self.in_queue, self.out_queue, self.signal_queue))

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


def run(gpuids, q):
    # scan all files under img_path
    names = [f for f in encoded_test_a.keys()]

    # init scheduler
    x = Scheduler(gpuids, q)

    # start processing and wait for complete
    return x.start(names)


def bleu(out_list):
    from nltk.translate.bleu_score import sentence_bleu

    annotations_path = os.path.join(test_a_folder, test_a_annotations_filename)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    score_list = []
    for item in out_list:
        image_name = item['image_name']
        candidate = item['candidate']
        image_hash = int(
            int(hashlib.sha256(image_name.split('.')[0].encode('utf-8')).hexdigest(), 16) % sys.maxsize)
        reference = [anno['caption'].split() for anno in annotations['annotations'] if
                     anno['image_id'] == image_hash]
        score = sentence_bleu(reference, candidate)
        score_list.append(score)
        # print('woker', self._gpuid, ' image_name ', image_name, " predicted as candidate", candidate)
        # print('score: {}, remaining tasks: {} '.format(score, self.in_queue.qsize()))
    return np.sum(score_list), np.mean(score_list)


SENTINEL = 1


def listener(q):
    pbar = tqdm(total=30000)
    for item in iter(q.get, None):
        pbar.update()


if __name__ == "__main__":
    gpuids = ['0', '1', '2', '3']
    print(gpuids)

    manager = mp.Manager()
    q = manager.Queue()
    proc = mp.Process(target=listener, args=(q,))
    proc.start()

    out_queue = run(gpuids, q)
    out_list = []
    while out_queue.qsize() > 0:
        out_list.append(out_queue.get())

    with open("preds.p", "wb") as file:
        pickle.dump(out_list, file)

    total_score, avg_score = bleu(out_list)

    print('item number: ' + str(len(out_list)))
    print('total score: ' + str(total_score))
    if len(out_list) > 0:
        print('avg: ' + str(avg_score))

    q.put(None)
    proc.join()
