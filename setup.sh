#!/bin/bash

pip install -r requirements.txt
sudo apt install unzip

gunzip data/WN18/synsetid2name.tsv.gz
gunzip data/FB15k/mid2name.tsv.gz

wget http://nlp.stanford.edu/data/glove.6B.zip -O HypER/language_models/glove/glove.6B.zip
unzip HypER/language_models/glove/glove.6B.zip -d HypER/language_models/glove/
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -O HypER/language_models/fasttext/cc.en.300.bin.gz
# gunzip -c HypER/language_models/fasttext/cc.en.300.bin.gz > HypER/language_models/fasttext/cc.en.300.bin

python HypER/language_models/attribute_mapper.py
python HypER/language_models/language_model_manager.py
