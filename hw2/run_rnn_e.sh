#!/bin/bash
# $1: input file
# $2: input tree file
# $3: output file

DIR="$(dirname $0)"

python3 "$DIR/rvnn/answer.py" --testing_data_file=$2 --output_file=$3
