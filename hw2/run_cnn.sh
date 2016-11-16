#!/bin/bash
# $1: input file
# $2: output file

DIR="$(dirname $0)"

python3 "$DIR/cnn/answer.py" --testing_data_file=$1 --output_file=$2
