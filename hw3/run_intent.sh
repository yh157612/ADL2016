#!/bin/bash
# $1: input file
# $2: answer file

DIR="$(dirname $0)"

python3 "$DIR/test.py" --test_data_file=$1 --output_tag_file=/dev/null --output_intent_file=$2
