# std
import os
import sys
import pickle
import logging
from datetime import datetime

# 3rd party
import fasttext
import bcolz
import numpy as np

# internal
try:
    import attribute_mapper as am
except ImportError as e:
    from language_models import attribute_mapper as am

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(root, dirname, filename) if filename else os.path.join(root, filename)
    return path


def save_language_model(language_model_data, language_model_version, language_model_size, language_model_dimension):
    logger.info(f'Saving Glove language model version {language_model_version} ...')

    words = []
    word2idx = {}
    err = 0

    language_model_file = f'{language_model_version}.dat'
    dirname = 'HypER/language_models/glove'
    language_model = get_path(language_model_file, dirname)
    vectors = bcolz.carray(np.zeros(1), rootdir=language_model, mode='w')

    logger.info('Populating language model file ...')

    path = get_path(language_model_data, dirname)
    with open(path, 'rb') as f:
        for idx, l in enumerate(f):
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx

            try:
                vect = np.array(line[1:]).astype(np.float)
                if vect.size != language_model_dimension:
                    vect = np.random.randn(language_model_dimension) * np.sqrt(1 / (language_model_dimension - 1))
            except ValueError as e:
                err += 1
                vect = np.random.randn(language_model_dimension) * np.sqrt(1 / (language_model_dimension - 1))
            finally:
                vectors.append(vect)

    logger.info('Populating language model file complete!')
    logger.info('Finalising language model file ...')

    vectors = bcolz.carray(vectors[1:].reshape((language_model_size, language_model_dimension)),
                           rootdir=language_model,
                           mode='w')
    vectors.flush()

    logger.info('Finalising language model file complete!')
    logger.info('Saving language model words ...')

    filename = f'{language_model_version}_words.pkl'
    path = get_path(filename, dirname)

    with open(path, 'wb') as language_model_words:
        pickle.dump(words, language_model_words)

    logger.info('Saving language model words complete!')
    logger.info('Saving language model word ids ...')

    filename = f'{language_model_version}_idx.pkl'
    path = get_path(filename, dirname)

    with open(path, 'wb') as language_model_ids:
        pickle.dump(word2idx, language_model_ids)

    logger.info('Saving language model word ids complete!')
    logger.info(f'Saving Glove language model version {language_model_version} complete!')


def load_glove(language_model_version):
    logger.info(f'Loading Glove language model ...')

    dirname = 'HypER/language_models/glove'

    filename = f'{language_model_version}.dat'
    path = get_path(filename, dirname)

    vectors = bcolz.open(path)[:]

    filename = f'{language_model_version}_words.pkl'
    path = get_path(filename, dirname)

    with open(path, 'rb') as wordsfile:
        words = pickle.load(wordsfile)

    filename = f'{language_model_version}_idx.pkl'
    path = get_path(filename, dirname)

    with open(path, 'rb') as word_ids:
        word2idx = pickle.load(word_ids)

    glove = {w: vectors[word2idx[w]] for w in words}

    logger.info(f'Loading Glove language model complete!')

    return glove


def load_fastext():
    logger.info(f'Loading Fasttext language model ...')

    language_model_name = 'cc.en.300.bin'
    dirname = 'HypER/language_models/fasttext'
    path = get_path(language_model_name, dirname)

    language_model = fasttext.load_model(path)

    logger.info(f'Loading Fasttext language model complete!')

    return language_model


def load_language_model(language_model_name, language_model_version, knowledge_graph):
    logger.info(f'Loading {language_model_name} language model version {language_model_version} and entity IDs map ...')

    if language_model_name == 'Fasttext':
        language_model = load_fastext()
    else:
        language_model_version = language_model_version
        language_model = load_glove(language_model_version)

    entity2idx = am.load_map(knowledge_graph)

    logger.info(f'Loading {language_model_name} language model version {language_model_version} '
                f'and entity IDs map complete!')

    return language_model, entity2idx


if __name__ == "__main__":
    logger.info('START!')

    language_model_name = 'Glove'
    language_model_version = '6B.200'
    language_model_data_map = {'6B.200':'glove.6B.200d.txt', 'twitter.27B.200': 'glove.twitter.27B.200d.txt'}
    language_model_size_map = {'6B.200': 400000, 'twitter.27B.200': 1193514}
    language_model_dimension_map = {'Glove': 200, 'Fasttext': 300}
    language_model_data = language_model_data_map[language_model_version]
    language_model_size = language_model_size_map[language_model_version]
    language_model_dimension = language_model_dimension_map[language_model_name]

    save_language_model(language_model_data, language_model_version, language_model_size, language_model_dimension)
    glove = load_glove(language_model_version)

    knowledge_graph = 'WN18'

    language_model, entity2idx = load_language_model(language_model_name, language_model_version, knowledge_graph)

    logger.info('DONE!')
