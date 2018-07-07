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


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    extract('train', 'ai_challenger_caption_train_20170902', 'ai_challenger_caption_train_20170902',
            '')

    extract('valid', 'ai_challenger_caption_validation_20170910', '',
            '')

    extract('test_a', 'ai_challenger_caption_test_a_20180103', 'scene_test_a_images_20180103',
            'scene_test_a_annotations_20180103.json')

    extract('test_b', 'ai_challenger_caption_test_b_20180103', 'scene_test_b_images_20180103',
            'scene_test_b_annotations_20180103.json')
