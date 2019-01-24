#!/usr/bin/env python
"""Implementation of Matrix Factorization with tensorflow.
Reference: Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009).
"""

import tensorflow as tf
import time
import numpy as np

from utils.evaluation.RatingMetrics import *
class MF_manu_dire_neg_pos():
    def __init__(self, sess, num_user, num_item, num_dire,learning_rate=0.001, reg_rate=0.01, epoch=500, batch_size=128,
                 show_time=False, T=2, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.num_dir = num_dire
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("MF.")

    def build_network(self, num_factor=30):

        # model dependent arguments
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.dire_pos_num = tf.placeholder(dtype=tf.int32, shape=[None], name='dire_pos_num')
        self.dire_neg_num = tf.placeholder(dtype=tf.int32, shape=[None], name='dire_neg_num')
        self.y = tf.placeholder("float", [None], name='rating')

        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=0.01))
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=0.01))

        self.B_U = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        self.B_I = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
        self.B_Dire_pos = tf.Variable(tf.zeros([self.num_dir]))
        self.B_Dire_neg = tf.Variable(tf.zeros([self.num_dir]))
        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        user_bias = tf.nn.embedding_lookup(self.B_U, self.user_id)
        item_bias = tf.nn.embedding_lookup(self.B_I, self.item_id)
        dire_pos_bias = tf.nn.embedding_lookup(self.B_Dire_pos, self.dire_pos_num)
        dire_neg_bias = tf.nn.embedding_lookup(self.B_Dire_neg, self.dire_neg_num)
        #self.pred_rating = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1) + user_bias + item_bias
        #self.pred_rating = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1) + dire_pos_bias + dire_neg_bias
        #self.pred_rating = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1) + dire_pos_bias + dire_neg_bias
        self.pred_rating = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1) + dire_pos_bias
        #self.pred_rating = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1)

    def train(self, train_data):
        self.num_training = len(train_data[0])
        total_batch = int(self.num_training/self.batch_size)

        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(np.array(train_data[0])[idxs])
        item_random = list(np.array(train_data[1])[idxs])
        rating_random = list(np.array(train_data[2])[idxs])
        dire_pos_random = list(np.array(train_data[3])[idxs])
        dire_neg_random = list(np.array(train_data[4])[idxs])
        # train
        for i in range(total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_dire_pos = dire_pos_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_dire_neg = dire_neg_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.dire_pos_num: batch_dire_pos,
                                                                            self.dire_neg_num: batch_dire_neg,
                                                                            self.y: batch_rating
                                                                            })
            if i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)) + "\n")
                if self.show_time:
                    print("one iteration: %s seconds." % (time.time() - start_time) + "\n")

    def test(self, test_data):
        error = 0
        error_mae = 0
        lens = len(test_data[0])
        for i in range(lens):
            pred_rating_test = self.predict([test_data[0][i]], [test_data[1][i]], [test_data[3][i]], [test_data[4][i]])
            error += (float(test_data[2][i]) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data[2][i]) - pred_rating_test))
        print("RMSE:" + str(RMSE(error, len(test_data[2]))[0]) + "; MAE:" + str(MAE(error_mae, len(test_data[2]))[0]))

    def execute(self, train_data, test_data):

        self.pred_rating += np.mean(train_data[2])
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + self.reg_rate * (
        tf.nn.l2_loss(self.B_Dire_neg) + tf.nn.l2_loss(self.B_Dire_pos) + tf.nn.l2_loss(self.B_I) + tf.nn.l2_loss(self.B_U) + tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q))
        # tf.norm(self.B_I) +  tf.norm(self.B_U) + tf.norm(self.P) +  tf.norm(self.Q))
        # tf.reduce_sum(tf.square(P))
        # tf.reduce_sum(tf.multiply(P,P))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            print("Epoch: %04d;" % (epoch))
            if (epoch) % self.T == 0:
                self.test(test_data)
            self.train(train_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id,dir_pos,dir_num_neg):
        return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id,self.dire_pos_num:dir_pos,\
                                                            self.dire_neg_num:dir_num_neg})[0]