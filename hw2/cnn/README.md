# Convolutional Neural Network (CNN) for Sentiment Analysis

> ADL Homework 2 Bonus 2

## Reference

- http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
- https://github.com/dennybritz/cnn-text-classification-tf

## How to use

### Training

```
$ python3 train.py --positive_data_file=/path/to/training_data.pos \
--negative_data_file=/path/to/training_data.neg
```

Models (checkpoint files) will be saved in folder `runs`.

### Predicting

Using your own model:

```
$ python3 answer.py --testing_data_file=/path/to/testing_data.txt \
--output_file=/path/to/output.txt \
--vocab_file=./runs/xxx/vocab \
--checkpoint_file=./runs/xxx/checkpoints/model-yyy
```

Using the pre-trained model:

```
$ python3 answer.py --testing_data_file=/path/to/testing_data.txt \
--output_file=/path/to/output.txt
```