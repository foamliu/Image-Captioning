import os
import zipfile
import bz2


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(package):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')

def extract_embedding():
    embedding_path = 'data/sgns.merge.word.bz2'
    with bz2.open(embedding_path, mode='rb') as f:
        for line_b in f:
            line = line_b.decode('utf-8')





if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract('ai_challenger_caption_train_20170902')

    extract('ai_challenger_caption_validation_20170910')

    extract('ai_challenger_caption_test_a_20180103')

    extract('ai_challenger_caption_test_b_20180103')


