#!/bin/bash

gunzip ./data/FB15k/mid2name.tsv.gz
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip -O HypER/language_models/glove/glove.twitter.27B.zip
sudo apt install unzip
unzip HypER/language_models/glove/glove.twitter.27B.zip -d HypER/language_models/glove/
python HypER/language_models/attribute_mapper.py
