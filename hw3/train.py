import os
import sys
import time

import numpy as np
import tensorflow as tf

from utils import prepare_train_data, split_data, save_vocabulary
from rnn import RNNModel


tf.app.flags.DEFINE_string('train_data_file', os.path.join(sys.path[0], 'data/atis.train.w-intent.iob'),
                           'Path to the training data file.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size (in sentence).')
tf.app.flags.DEFINE_integer('hidden_size', 128, 'Size of vectors in RNN cells.')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'Size of the word embedding.')
tf.app.flags.DEFINE_integer('vocab_size', 10000, 'Max vocabulary size.')
tf.app.flags.DEFINE_integer('num_epoch', 100, 'Number of epochs to train.')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep cell input and output probability.')
FLAGS = tf.app.flags.FLAGS


def zero_padding(l):
    max_len = max([len(a) for a in l])
    mat = np.zeros([len(l), max_len])
    for i, a in enumerate(l):
        mat[i, :len(a)] = a
    return mat


def batch_generator(sentence_list, tags_list, intent_list):
    batch_size = FLAGS.batch_size
    for i in range(0, len(sentence_list), batch_size):
        if i + batch_size > len(sentence_list):
            break
        sentence_batch = sentence_list[i:i + batch_size]
        tags_batch = tags_list[i:i + batch_size]
        intent_batch = intent_list[i:i + batch_size]
        length_batch = [len(a) for a in sentence_batch]
        sentence_batch = zero_padding(sentence_batch)
        tags_batch = zero_padding(tags_batch)
        yield sentence_batch, length_batch, tags_batch, intent_batch


def main(_):
    all_sentence, all_tags, all_intent, vocab, dictionary, tags_list, tags_dict, intent_list, intent_dict = prepare_train_data(FLAGS.train_data_file, FLAGS.vocab_size)
    train_data, dev_data = split_data(all_sentence, all_tags, all_intent)
    # train_sentence, train_tags, train_intent = train_data
    # dev_sentence, dev_tags, dev_intent = dev_data

    output_path = os.path.join(sys.path[0], 'runs', str(int(time.time())))
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    os.makedirs(checkpoint_dir, mode=0o755, exist_ok=True)

    save_vocabulary(os.path.join(output_path, 'sentence_vocab'), vocab)
    save_vocabulary(os.path.join(output_path, 'tag_vocab'), tags_list)
    save_vocabulary(os.path.join(output_path, 'intent_vocab'), intent_list)

    model = RNNModel(hidden_size=FLAGS.hidden_size,
                     embed_size=FLAGS.embedding_size,
                     source_vocab_size=len(vocab),
                     tag_vocab_size=len(tags_list),
                     intent_vocab_size=len(intent_list))

    with tf.Session(graph=model.graph) as sess:
        sess.run(tf.initialize_all_variables())

        step = 1
        avg_tag_loss = 0
        avg_intent_loss = 0
        for epoch in range(FLAGS.num_epoch):
            batch_gen = batch_generator(*train_data)
            for sentence_batch, length_batch, tags_batch, intent_batch in batch_gen:
                _, tag_loss, intent_loss = sess.run([model.train_op, model.tag_loss, model.intent_loss], feed_dict={
                    model.input_x: sentence_batch,
                    model.input_len: length_batch,
                    model.input_tag: tags_batch,
                    model.input_intent: intent_batch,
                    model.keep_prob: FLAGS.dropout_keep_prob
                })
                avg_tag_loss += tag_loss
                avg_intent_loss += intent_loss
                if step % 20 == 0:
                    avg_tag_loss /= 20
                    avg_intent_loss /= 20
                    print('Step', step, 'Tag loss', tag_loss, 'Intent loss', intent_loss)
                    avg_tag_loss = 0
                    avg_intent_loss = 0
                step += 1

            correct_tag, total_tag = 0, 0
            correct_intent, total_intent = 0, 0
            for sentence, tags, intent in zip(*dev_data):
                predict_tags, predict_intent = sess.run([model.output_tag, model.output_intent], feed_dict={
                    model.input_x: [sentence],
                    model.input_len: [len(sentence)],
                    model.keep_prob: 1.0
                })
                for tag1, tag2 in zip(tags, predict_tags[0]):
                    if tag1 == tag2:
                        correct_tag += 1
                    total_tag += 1
                if intent == predict_intent[0]:
                    correct_intent += 1
                total_intent += 1
            tag_accuracy = correct_tag / total_tag
            intent_accuracy = correct_intent / total_intent
            print('[Validation]', 'tag acc =', tag_accuracy, ', intent acc =', intent_accuracy, '\n')
            model.saver.save(sess, os.path.join(checkpoint_dir, '{}_{:.4f}_{:.4f}.ckpt'.format(epoch, tag_accuracy, intent_accuracy)))


if __name__ == '__main__':
    tf.app.run()
