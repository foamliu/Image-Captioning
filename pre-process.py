import os
import zipfile


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(usage, package, image_path, json_path):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def extract_test(usage, package, image_path, json_path):
    filename = 'data/{}.zip'.format(package)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract('train', 'ai_challenger_caption_train_20170902', '',
            '')

    extract('valid', 'ai_challenger_caption_validation_20170910', '',
            '')
