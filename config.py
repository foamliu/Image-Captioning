import numpy as np
import os
from tqdm import tqdm

img_rows, img_cols = 224, 224
channel = 3
batch_size = 64
epochs = 10000
patience = 50
num_train_samples = 210000 * 5 / 10
num_valid_samples = 30000 * 5 / 10
embedding_size = 300
vocab_size = 332647 + 3
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
test_a_image_folder = os.path.join(test_a_folder, 'caption_test_a_images_20180103')
test_b_image_folder = os.path.join(test_b_folder, 'caption_test_b_images_20180103')
train_annotations_filename = 'caption_train_annotations_20170902.json'
valid_annotations_filename = 'caption_validation_annotations_20170910.json'

print('loading word embedding...')
from gensim.models import KeyedVectors

zh_model = KeyedVectors.load_word2vec_format('data/wiki.zh.vec')
embedding_matrix = np.zeros((vocab_size, embedding_size))
for index, word in tqdm(enumerate(zh_model.vocab)):
    embedding_matrix[index, :] = zh_model[word]
vocab = zh_model.vocab
idx2word = list(vocab.keys())
start_word = '<START>'
stop_word = '<EOS>'
unknown_word = '<UNK>'
idx2word.append(start_word)
idx2word.append(stop_word)
idx2word.append(unknown_word)
word2idx = dict(zip(idx2word, range(vocab_size)))

embedding_matrix[vocab_size-3] = np.random.rand(embedding_size,)
embedding_matrix[vocab_size-2] = np.random.rand(embedding_size,)
embedding_matrix[vocab_size-1] = np.random.rand(embedding_size,)

