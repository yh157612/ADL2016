#!/bin/bash
# $1: corpus path
# $2: output directory

mkdir $2
mkdir tmp

python3 word2vec_optimized.py --train_data=$1 --eval_data=questions-ptt.txt --save_path=./tmp/vec.txt
python2 filterVocab.py pttVocab.txt < ./tmp/vec.txt > $2/filter_vec.txt