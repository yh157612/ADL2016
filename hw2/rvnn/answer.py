"""ADL Homework 2
Recursive Neural Network (RvNN)
Sentiment Analysis
"""
import os
import sys

import tensorflow as tf

from tree_parser import load_trees
from rvnn import tree_to_matrix


def main(FLAGS):
    # Read testing data files
    test_data = load_trees(FLAGS.testing_data_file)

    vocab_path = os.path.join(sys.path[0], 'runs/1479118465/vocab')
    checkpoint_path = os.path.join(sys.path[0], 'runs/1479118465/checkpoints/22000.ckpt')

    # Build dictionary
    with open(vocab_path) as f:
        word_list = f.read().split()
    dictionary = {word: i for i, word in enumerate(word_list)}
    vocab_size = len(dictionary)
    print('Vocabulary size:', vocab_size)

    # Build graph
    all_predictions = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_path))
        saver.restore(sess, checkpoint_path)

        input_x = tf.get_default_graph().get_tensor_by_name('input_x:0')
        predict = tf.get_default_graph().get_tensor_by_name('prediction:0')

        for tree in test_data:
            feed_dict = {input_x: tree_to_matrix(tree, dictionary)}
            prediction = sess.run(predict, feed_dict=feed_dict)
            all_predictions.append(str(prediction))

    # Save answer file
    output_path = os.path.abspath(FLAGS.output_file)
    os.makedirs(os.path.dirname(output_path), mode=0o755, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(all_predictions))


if __name__ == '__main__':
    tf.flags.DEFINE_string("testing_data_file", "", "Data source for the testing data.")
    tf.flags.DEFINE_string("output_file", "", "Output prediction file path")
    tf.flags.FLAGS._parse_flags()
    main(tf.flags.FLAGS)
