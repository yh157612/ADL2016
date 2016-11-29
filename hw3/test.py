import os
import sys

import tensorflow as tf

from utils import prepare_test_data, load_vocabulary


tf.app.flags.DEFINE_string('test_data_file', os.path.join(sys.path[0], 'data/atis.test.iob'),
                           'Path to the testing data file.')
tf.app.flags.DEFINE_string('vocab_dir', os.path.join(sys.path[0], 'runs/1480320045'),
                           'Path to the vocabulary directory.')
tf.app.flags.DEFINE_string('checkpoint_file', os.path.join(sys.path[0], 'runs/1480320045/checkpoints/91_0.9841_0.9759.ckpt'),
                           'Path to the checkpoint file.')
tf.app.flags.DEFINE_string('output_tag_file', './answer.tag.txt',
                           'Path to the output tag file.')
tf.app.flags.DEFINE_string('output_intent_file', './answer.intent.txt',
                           'Path to the output intent file.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    vocab, dictionary = load_vocabulary(os.path.join(FLAGS.vocab_dir, 'sentence_vocab'))
    tags_list, tags_dict = load_vocabulary(os.path.join(FLAGS.vocab_dir, 'tag_vocab'))
    intent_list, intent_dict = load_vocabulary(os.path.join(FLAGS.vocab_dir, 'intent_vocab'))
    all_sentence = prepare_test_data(FLAGS.test_data_file, dictionary)

    all_tags = []
    all_intent = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(FLAGS.checkpoint_file))
        saver.restore(sess, FLAGS.checkpoint_file)

        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name('input_x:0')
        input_len = graph.get_tensor_by_name('input_len:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        output_tag = graph.get_tensor_by_name('output_tag:0')
        output_intent = graph.get_tensor_by_name('output_intent:0')

        for sentence in all_sentence:
            predict_tags, predict_intent = sess.run([output_tag, output_intent], feed_dict={
                input_x: [sentence],
                input_len: [len(sentence)],
                keep_prob: 1.0
            })
            all_tags.append(predict_tags[0])
            all_intent.append(predict_intent[0])

    all_tags = [['O'] + [tags_list[i] for i in tags] for tags in all_tags]
    all_intent = [intent_list[i] for i in all_intent]
    with open(FLAGS.output_tag_file, 'w') as f:
        f.write('\n'.join([' '.join(tags) for tags in all_tags]))
    with open(FLAGS.output_intent_file, 'w') as f:
        f.write('\n'.join(all_intent))


if __name__ == '__main__':
    tf.app.run()
