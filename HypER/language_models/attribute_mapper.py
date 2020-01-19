# std
import os
import sys
import csv
import logging
import pickle as pkl

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


def save_map(knowledge_graph='FB15k', attribute='entity', entity_id_2_name_file='mid2name.tsv', delimiter='\t'):
    logger.info(f'Saving {attribute} map for {knowledge_graph} ...')

    word2idx = {}

    dirname = f'data/{knowledge_graph}'
    filename = entity_id_2_name_file
    path = get_path(filename, dirname)

    with open(path, 'r', encoding='utf-8') as entityfile:
        tsv_reader = csv.reader(entityfile, delimiter=delimiter)

        for line in tsv_reader:
            idx = line[0]
            entity = word2idx.get(idx)
            if not entity:
                entity = line[1]
                word2idx[idx] = entity

    filename = f'{knowledge_graph}_{attribute}_map.pkl'
    dirname = f'HypER/language_models/{knowledge_graph}'
    path = get_path(filename, dirname)

    with open(path, 'wb') as writepickle:
        pkl.dump(word2idx, writepickle)

    logger.info(f'Successfully saved {attribute} map for {knowledge_graph}!')


def load_map(knowledge_graph, attribute='entity'):
    logger.info(f'Loading {attribute} ids map ...')

    entityids_map = f'{knowledge_graph}_{attribute}_map.pkl'
    dirname = f'HypER/language_models/{knowledge_graph}'
    path = get_path(entityids_map, dirname)

    with open(path, 'rb') as attribute_ids_map:
        word2idx = pkl.load(attribute_ids_map)

    logger.info(f'Successfully loaded {attribute} ids map!')

    return word2idx


if __name__ == '__main__':
    logger.info('START!')

    knowledge_graph = 'WN18'
    attribute = 'entity'
    entity_id_2_name = 'synsetid2name.tsv'

    save_map(knowledge_graph, attribute, entity_id_2_name, '\t')
    fb15k_entity_map = load_map(knowledge_graph, attribute)

    logger.info('DONE!')
