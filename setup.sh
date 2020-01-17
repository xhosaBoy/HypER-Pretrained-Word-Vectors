#!/bin/bash

wget https://drive.google.com/drive/u/0/folders/13-_KyI8vhW2LztXMdR-JuFqKFSl9UTaw -O /tmp/mid2name.tsv
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip -O ./language_models/glove/glove.twitter.27B.zip
sudo apt install unzip
unzip language_models/glove/glove.twitter.27B.zip