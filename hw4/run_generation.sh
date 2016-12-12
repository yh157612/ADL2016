#!/bin/bash
# $1: input file
# $2: answer file

DIR="$(dirname $0)"

if [ -f $2 ]; then
    rm -f $2
fi

python3 "$DIR/nlg/translate.py" --decode < $1 > $2
