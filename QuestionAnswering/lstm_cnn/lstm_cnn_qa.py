
from QuestionAnswering.lstm_cnn import qa_data
import tensorflow as tf
import random


class CNN_QA(object):

    def __init__(self, sequence_length, batch_size, embedding_size, epoch, filter_sizes, num_filters):

        self.sequence_length, self.batch_size, self.embedding_size, self.epoch, self.filter_sizes, self.num_filters = \
            sequence_length, batch_size, embedding_size, epoch, filter_sizes, num_filters

        self.q, self.qp = self.load_data()
        self.qn = random.shuffle(self.qp)
        self.dropout_keep_prob = 0.2
        self.margin = 0.02


        x_qp = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length, self.embedding_size])
        x_qn = tf.plecaholder(tf.float32, [self.batch_size, self.sequence_length, self.embedding_size])
        x_a = tf.placeholder(tf.float32, [self.batch_size, self.sequence_length, self.embedding_size])

        qp_conv = self.conv(x_qp)
        qn_conv = self.conv(x_qn)
        a_conv = self.conv(x_a)

        cosin_q_qp = self.coscin(qp_conv, a_conv)
        cosin_q_qn = self.coscin(qn_conv, a_conv)
        zeros = tf.constant(0, shape=[self.batch_size])
        margins = tf.constant(self.margin, shape=[self.batch_size])

        losses = tf.maximum(tf.substract(margins, tf.sunstract(cosin_q_qp, cosin_q_qn)))
        self.loss = tf.reduce_sum(losses)


    def conv(self, tensor):

        with tf.variable_scope('conv_shared'):
            pooled = []
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.float32, initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                conv = tf.nn.conv2d(tensor, W, strids=[1, 1, 1, 1], padding='VALID')

                b = tf.Variable(tf.float32, initializer=tf.truncates_normal([self.num_filters], stddev=0.1))

                h = tf.nn.relu(tf.nn.bias_add(conv, b))

                output = tf.nn.max_pool(h, [1, self.sequence_length-self.filter_sizes+1, 1, 1],
                                        padding='VALID',
                                        name='pooling')
                pooled.append(output)


            pooled = tf.reshape(tf.concat(pooled, 3), [-1, len(self.filter_sizes)*self.num_filters])
            pooled = tf.nn.dropout(pooled, self.dropout_keep_prob)

            return pooled

    def coscin(self, v1, v2):
        l1 = tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1))
        l2 = tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1))
        a = tf.reduce_sum(tf.multiply(v1, v2), 1)

        similarity = tf.div(a, tf.multiply(l1, l2))

        return tf.clip_by_vlaue(similarity, 1e-5, 0.9999)


