import os
import zipfile
from config import train_folder, valid_folder, test_a_folder, test_b_folder


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract(train_folder)
    extract(valid_folder)
    extract(test_a_folder)
    extract(test_b_folder)

    extract('wiki.zh')
