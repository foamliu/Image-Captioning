img_rows, img_cols = 256, 256
channel = 3
batch_size = 128
epochs = 10000
patience = 50
num_train_samples = 210000
num_valid_samples = 30000
embedding_size = 300
vocab_size = 332647
max_cap_len = 40

rnn_type = 'gru'
bidirectional_rnn = False
hidden_size = 512
rnn_dropout_rate = 0.5
rnn_layers = 2

train_folder = 'data/ai_challenger_caption_train_20170902'
valid_folder = 'data/ai_challenger_caption_validation_20170910'
test_a_folder = 'data/ai_challenger_caption_test_a_20180103'
test_b_folder = 'data/ai_challenger_caption_test_b_20180103'
train_annotations_filename = 'caption_train_annotations_20170902.json'
valid_annotations_filename = 'caption_validation_annotations_20170910.json'

from gensim.models import KeyedVectors
zh_model = KeyedVectors.load_word2vec_format('data/wiki.zh.vec')
