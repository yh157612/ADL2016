import tensorflow as tf


class RNNModel(object):

    def __init__(self, hidden_size, embed_size, source_vocab_size, tag_vocab_size, intent_vocab_size):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.input_x = tf.placeholder(tf.int32, [None, None], 'input_x')  # shape=[batch_size, max_seq_length]
            self.input_len = tf.placeholder(tf.int32, [None], 'input_len')
            self.input_tag = tf.placeholder(tf.int32, [None, None], 'input_tag')
            self.input_intent = tf.placeholder(tf.int32, [None], 'input_intent')
            self.keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')

            batch_size, max_seq_len = tf.unpack(tf.shape(self.input_x))

            W_tag = tf.get_variable('W_tag', [hidden_size, tag_vocab_size], tf.float32)
            b_tag = tf.get_variable('b_tag', [tag_vocab_size], tf.float32)
            W_intent = tf.get_variable('W_intent', [hidden_size, intent_vocab_size], tf.float32)
            b_intent = tf.get_variable('b_intent', [intent_vocab_size], tf.float32)

            with tf.device('/cpu:0'):
                embeddings = tf.get_variable('embeddings', [source_vocab_size, embed_size], tf.float32)
                input_lookup = tf.nn.embedding_lookup(embeddings, self.input_x)

            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.keep_prob, self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, input_lookup, self.input_len, dtype=tf.float32)

            # Calculate loss for tagging
            outputs_flat = tf.reshape(outputs, [-1, hidden_size])
            logits_flat = tf.matmul(outputs_flat, W_tag) + b_tag
            self.output_tag = tf.reshape(tf.argmax(logits_flat, dimension=1), [batch_size, max_seq_len], name='output_tag')
            tag_flat = tf.reshape(self.input_tag, [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, tag_flat)

            ranges = tf.tile(tf.reshape(tf.range(0, max_seq_len), [1, -1]), [batch_size, 1])
            len_col_vec = tf.reshape(self.input_len, [-1, 1])
            mask = tf.to_float(ranges < len_col_vec)
            mask_flat = tf.reshape(mask, [-1])

            masked_loss = mask_flat * losses
            self.tag_loss = tf.reduce_sum(masked_loss) / tf.to_float(tf.reduce_sum(self.input_len))

            # Calculate loss for intent
            logits = tf.matmul(state, W_intent) + b_intent
            self.output_intent = tf.argmax(logits, dimension=1, name='output_intent')
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.input_intent)
            self.intent_loss = tf.reduce_mean(losses)

            self.loss = self.tag_loss + self.intent_loss
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
            self.saver = tf.train.Saver(max_to_keep=0)
