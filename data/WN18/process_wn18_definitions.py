# std
import os
import sys
import re
import csv
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    logger.debug(f'root: {root}')
    path = os.path.join(root, dirname, filename) if dirname else os.path.join(root, filename)
    return path


def transform_definitions(filename, dirname, delimiter='\t'):
    logger.info(f'Transforming {filename} to entity to ID map ...')

    word2idx = {}

    path = get_path(filename, dirname)

    with open(path) as wn18_definitions:
        tsv_reader = csv.reader(wn18_definitions,  delimiter=delimiter)

        for line in tsv_reader:
            idx = line[0]
            entity = word2idx.get(idx)
            if not entity:
                entity = line[1]
                pattern = re.compile(r'^__([a-zA-Z0-9\'\._/-]*)_([A-Z]{2})_([0-9])')
                entity = pattern.search(entity).group(1).replace('_', ' ')
                word2idx[idx] = entity

    logger.info(f'Transforming {filename} to entity to ID map complete!')

    return word2idx


def write_word2idx(word2idx, filename, dirname, delimiter='\t'):
    logger.info(f'Writing {filename} to entity to ID map ...')

    path = get_path(filename, dirname)
    with open(path, 'w') as wn18_tsv:
        csv_writer = csv.writer(wn18_tsv, delimiter=delimiter)

        for id, entity in word2idx.items():
            csv_writer.writerow([id, entity])

    logger.info(f'Writing {filename} to entity to ID map complete!')


if __name__ == '__main__':
    logger.info('START!')

    filename = 'wordnet-mlj12-definitions.txt'
    dirname = 'data/WN18'

    word2idx = transform_definitions(filename, dirname, '\t')

    filename = 'synsetid2name.tsv'
    write_word2idx(word2idx, filename, dirname)

    logger.info('DONE!')
