#!/bin/bash
# $1: corpus path
# $2: output directory

mkdir $2
mkdir tmp

python3 word2vec_optimized.py --train_data=$1 --save_path=./tmp/word2vec.txt
python2 filterVocab.py fullVocab.txt < ./tmp/word2vec.txt > $2/filter_word2vec.txt

python3 glove.py $1 ./tmp/glove.txt
python2 filterVocab.py fullVocab.txt < ./tmp/glove.txt > $2/filter_glove.txt