import tensorflow as tf


def combine_word_vectors(left, right, weights, bias, name=None):
    with tf.name_scope(name, 'CombineWordVectors',
                       [left, right, weights, bias]) as scope:
        left = tf.reshape(left, [1, -1], name='left')
        right = tf.reshape(right, [1, -1], name='right')
        weights = tf.convert_to_tensor(weights, name='weights')
        bias = tf.reshape(bias, [1, -1], name='bias')
        concat_vector = tf.concat(1, [left, right])
        return tf.nn.relu(tf.matmul(concat_vector, weights) + bias,
                          name=scope)


def tree_to_matrix(tree, dictionary):
    mat = []

    def build_mat(node):
        if not node.children:
            index = (dictionary[node.content]
                     if node.content in dictionary
                     else 0)
            return ~index  # -index - 1
        else:
            l_index = build_mat(node.children[0])
            r_index = build_mat(node.children[1])
            mat.append([l_index, r_index])
            return len(mat) - 1

    build_mat(tree)
    return mat
