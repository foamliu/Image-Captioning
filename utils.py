import multiprocessing

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from config import vocab_size, embedding_size, start_word, stop_word, unknown_word


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def sparse_loss(y_true, y_pred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def load_word_index_converts():
    from gensim.models import KeyedVectors
    zh_model = KeyedVectors.load_word2vec_format('data/wiki.zh.vec')
    vocab = zh_model.vocab
    idx2word = list(vocab.keys())
    idx2word.append(start_word)
    idx2word.append(stop_word)
    idx2word.append(unknown_word)
    word2idx = dict(zip(idx2word, range(vocab_size)))
    return idx2word, word2idx


def load_word_embedding():
    from gensim.models import KeyedVectors
    from tqdm import tqdm
    print('loading word embedding...')
    zh_model = KeyedVectors.load_word2vec_format('data/wiki.zh.vec')
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for index, word in tqdm(enumerate(zh_model.vocab)):
        embedding_matrix[index, :] = zh_model[word]

    np.random.seed(1)
    embedding_matrix[vocab_size - 3] = np.random.rand(embedding_size, )
    embedding_matrix[vocab_size - 2] = np.random.rand(embedding_size, )
    embedding_matrix[vocab_size - 1] = np.random.rand(embedding_size, )

    return
