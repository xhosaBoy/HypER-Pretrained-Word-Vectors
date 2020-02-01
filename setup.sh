#!/bin/bash

gunzip data/WN18/synsetid2name.tsv.gz
gunzip data/FB15k/mid2name.tsv.gz
wget http://nlp.stanford.edu/data/glove.6B.zip -O HypER/language_models/glove/glove.6B.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -O HypER/language_models/fasttext/cc.en.300.bin.gz
sudo apt install unzip
unzip HypER/language_models/glove/glove.twitter.27B.zip -d HypER/language_models/glove/
gunzip -c HypER/language_models/fasttext/cc.en.300.bin.gz > HypER/language_models/fasttext/cc.en.300.bin
pip install -r requirements.txt
python HypER/language_models/attribute_mapper.py
python HypER/language_models/language_model_manager.py
