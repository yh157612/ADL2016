"""ADL Homework 2
Recursive Neural Network (RvNN)
Sentiment Analysis
"""
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import tensorflow as tf

from rvnn import combine_word_vectors, tree_to_matrix
from tree_parser import load_trees


def main():
    # Read training data files
    pos_trees = load_trees('training_data.pos.tree')
    neg_trees = load_trees('training_data.neg.tree')
    pos_trees.sort(key=lambda it: len(it.words()))
    neg_trees.sort(key=lambda it: len(it.words()))
    pos_data = [(tree, 1) for tree in pos_trees]
    neg_data = [(tree, 0) for tree in neg_trees]

    # Split data for validation
    validate_data_size = int(0.1 * len(pos_data))
    train_data = pos_data[:-validate_data_size] + neg_data[:-validate_data_size]
    validate_data = pos_data[-validate_data_size:] + neg_data[-validate_data_size:]
    train_data_2 = (pos_data[-2 * validate_data_size:-validate_data_size] +
                    neg_data[-2 * validate_data_size:-validate_data_size])
    random.shuffle(train_data)
    print('Training data size:', len(train_data))
    print('Validate data size:', len(validate_data))

    output_dir = './runs/{}/'.format(int(datetime.now().timestamp()))
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, 0o755, exist_ok=True)

    # Build dictionary
    words = []
    for tree, _ in train_data:
        words.extend(tree.words())
    word_list = [word for word, count in Counter(words).most_common()]
    word_list.insert(0, '<unk>')
    dictionary = {word: i for i, word in enumerate(word_list)}
    vocab_size = len(dictionary)
    print('Vocabulary size:', vocab_size)
    with open(os.path.join(output_dir, 'vocab'), 'w') as f:
        f.write('\n'.join(word_list))

    # Parameters
    embed_size = 32
    learning_rate = 0.001
    epoch_to_train = 5

    # Build graph
    input_x = tf.placeholder(tf.int32, [None, 2], name='input_x')
    input_y = tf.placeholder(tf.int32, [])

    W = (0.5 * np.concatenate((np.eye(embed_size), np.eye(embed_size)), axis=0) +
         np.random.uniform(-0.001, 0.001, (2 * embed_size, embed_size)))

    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, embed_size], -0.001, 0.001), name='embeddings')
    combine_W = tf.Variable(W, name='combine_W', dtype=tf.float32)
    combine_b = tf.Variable(
        tf.random_uniform([1, embed_size], -0.001, 0.001), name='combine_b')
    classify_W = tf.get_variable('classify_W', [embed_size, 2])
    classify_b = tf.get_variable('classify_b', [1, 2])

    num_x_rows = tf.gather(tf.shape(input_x), 0)
    vectors = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                             clear_after_read=False, infer_shape=False)

    def body(pos, vectors):
        left_idx, right_idx = tf.unpack(tf.gather(input_x, pos))
        left_vec = tf.cond(
            left_idx < 0,
            lambda: tf.gather(embeddings, -left_idx - 1),
            lambda: vectors.read(left_idx))
        right_vec = tf.cond(
            right_idx < 0,
            lambda: tf.gather(embeddings, -right_idx - 1),
            lambda: vectors.read(right_idx))
        new_vec = combine_word_vectors(left_vec, right_vec, combine_W, combine_b)
        return pos + 1, vectors.write(pos, new_vec)

    def cond(pos, vectors):
        return pos < num_x_rows

    _, vectors_result = tf.while_loop(cond, body, [0, vectors], parallel_iterations=1)
    sentence_vector = vectors_result.read(num_x_rows - 1)
    unscaled_prob = tf.squeeze(tf.matmul(sentence_vector, classify_W) + classify_b)
    prediction = tf.argmax(unscaled_prob, 0, name='prediction')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_prob, input_y)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        step = 1
        avg_loss = 0
        for epoch in range(epoch_to_train):
            for tree, label in train_data:
                feed_dict = {
                    input_x: tree_to_matrix(tree, dictionary),
                    input_y: label
                }
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                avg_loss += loss_value
                if step % 100 == 0:
                    print('Step {:<5} Loss {:<30}'.format(step, avg_loss / 100))
                    avg_loss = 0
                if step % 2000 == 0:
                    output_path = os.path.join(checkpoint_dir, '{}.ckpt'.format(step))
                    save_path = saver.save(sess, output_path)
                    print('Model saved in file:', save_path)
                step += 1

            correct = 0
            for tree, label in train_data_2:
                feed_dict = {
                    input_x: tree_to_matrix(tree, dictionary)
                }
                predict_result = sess.run(prediction, feed_dict=feed_dict)
                if predict_result == label:
                    correct += 1
            print('Training set accuracy =', correct / len(train_data_2), '\n')

            correct = 0
            for tree, label in validate_data:
                feed_dict = {
                    input_x: tree_to_matrix(tree, dictionary)
                }
                predict_result = sess.run(prediction, feed_dict=feed_dict)
                if predict_result == label:
                    correct += 1
            print('Validation set accuracy =', correct / len(validate_data), '\n')


if __name__ == '__main__':
    main()
