#! /usr/bin/env python

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("testing_data_file", "", "Data source for the testing data.")
tf.flags.DEFINE_string("output_file", "", "Output prediction file path")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("vocab_file",
                       os.path.join(sys.path[0], "runs/1478796720/vocab"),
                       "Vocabulary file path from training run")
tf.flags.DEFINE_string("checkpoint_file",
                       os.path.join(sys.path[0], "runs/1478796720/checkpoints/model-14200"),
                       "Checkpoint file path from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
x_raw = data_helpers.load_testing_data(FLAGS.testing_data_file)

# Map data into vocabulary
vocab_path = FLAGS.vocab_file
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nAnswering...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.checkpoint_file
print('Checkpoint file:', checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Save the answers
output_path = os.path.abspath(FLAGS.output_file)
os.makedirs(os.path.dirname(output_path), mode=0o755, exist_ok=True)
with open(output_path, 'w') as f:
    f.write('\n'.join(all_predictions.astype(int).astype(str)))
