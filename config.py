import numpy as np
import os
from tqdm import tqdm

img_rows, img_cols = 224, 224
channel = 3
batch_size = 128
epochs = 10000
patience = 50
num_train_samples = 210000
num_valid_samples = 30000
embedding_size = 300
vocab_size = 332647
max_token_length = 40
num_image_features = 2048

rnn_type = 'gru'
bidirectional_rnn = False
hidden_size = 512
rnn_dropout_rate = 0.5
rnn_layers = 2
regularizer = 1e-8

train_folder = 'data/ai_challenger_caption_train_20170902'
valid_folder = 'data/ai_challenger_caption_validation_20170910'
test_a_folder = 'data/ai_challenger_caption_test_a_20180103'
test_b_folder = 'data/ai_challenger_caption_test_b_20180103'
train_image_folder = os.path.join(train_folder, 'caption_train_images_20170902')
valid_image_folder = os.path.join(valid_folder, 'caption_validation_images_20170910')
train_annotations_filename = 'caption_train_annotations_20170902.json'
valid_annotations_filename = 'caption_validation_annotations_20170910.json'

print('loading word embedding...')
from gensim.models import KeyedVectors

zh_model = KeyedVectors.load_word2vec_format('data/wiki.zh.vec')
embedding_matrix = np.zeros((vocab_size, embedding_size))
for index, word in tqdm(enumerate(zh_model.vocab)):
    embedding_matrix[index, :] = zh_model[word]
vocab = zh_model.vocab

