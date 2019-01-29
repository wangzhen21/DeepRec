#!/usr/bin/env python
"""Implementation of Matrix Factorization with tensorflow.
Reference: Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009).
Orginal Implementation:
"""
from __future__ import print_function
import tensorflow as tf
import time
from sklearn.metrics import mean_squared_error
import math
import datetime

from utils.evaluation.RatingMetrics import *
__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"

def outfile(file,outstr):
    with open(file, 'awb+') as f:
        f.write(outstr +"\n")
class NFM_dire():

    def __init__(self, sess, num_user, num_item, learning_rate = 0.01, reg_rate = 0.01, epoch = 500, batch_size = 128, show_time = True, T =1, display_step= 1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("NFM.")


    def build_network(self, feature_M, num_factor = 64, num_hidden = 128):

        # model dependent arguments
        self.train_features = tf.placeholder(tf.int32, shape=[None, None])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.dropout_keep = tf.placeholder(tf.float32)

        self.feature_embeddings = tf.Variable(tf.random_normal([feature_M, num_factor], mean=0.0, stddev=0.01))

        self.feature_bias = tf.Variable(tf.random_uniform([feature_M, 1], 0.0, 0.0))
        self.bias = tf.Variable(tf.constant(0.0))
        self.pred_weight = tf.Variable(np.random.normal(loc=0, scale= np.sqrt(2.0 / (num_factor + num_hidden)), size=(num_hidden, 1)),
                                                dtype=np.float32)

        nonzero_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, self.train_features)

        self.summed_features_embedding = tf.reduce_sum(nonzero_embeddings, 1)
        self.squared_summed_features_embedding = tf.square(self.summed_features_embedding)
        self.squared_features_embedding = tf.square(nonzero_embeddings)
        self.summed_squared_features_embedding = tf.reduce_sum(self.squared_features_embedding, 1)

        self.FM = 0.5 * tf.subtract( self.squared_summed_features_embedding, self.summed_squared_features_embedding)
        # if batch_norm:
        #     self.FM = self
        layer_1 = tf.layers.dense(inputs=self.FM, units=num_hidden,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.FM =  tf.matmul(tf.nn.dropout(layer_1, self.dropout_keep), self.pred_weight)



        bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)
        self.f_b = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_bias, self.train_features), 1)
        b = self.bias * tf.ones_like(self.y)
        self.pred_rating = tf.add_n([bilinear, self.f_b, b])

        self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.pred_rating)) \
                    + tf.contrib.layers.l2_regularizer(self.reg_rate)(self.feature_embeddings)


        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):


        print("data preparation finished.")
        return self


    def train(self, train_data):
        self.num_training = len(train_data['Y'])
        total_batch = int( self.num_training/ self.batch_size)

        rng_state = np.random.get_state()
        np.random.shuffle(train_data['Y'])
        np.random.set_state(rng_state)
        np.random.shuffle(train_data['X'])
        # train
        loss_list = []
        start_time = time.time()
        for i in range(total_batch):
            batch_y = train_data['Y'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_x = train_data['X'][i * self.batch_size:(i + 1) * self.batch_size]

            loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict={self.train_features: batch_x,
                                                                              self.y: batch_y,
                                                                              self.dropout_keep:0.8})
            loss_list.append(loss)
        print("cost= %.9f" % (np.mean(np.array(loss_list))))
        if self.show_time:
            print(" %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        # error = 0
        # error_mae = 0
        # test_set = list(test_data.keys())
        # for (u, i) in test_set:
        #     pred_rating_test = self.predict([u], [i])
        #     error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
        #     error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        num_example = len(test_data['Y'])
        feed_dict = {self.train_features: test_data['X'], self.y: test_data['Y'],self.dropout_keep: 1.0}
        predictions = self.sess.run((self.pred_rating), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(test_data['Y'], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

        print("RMSE:" + str(RMSE))
        outfile("../log/NFM_dire.log", self.starttime + "\t" + "RMSE\t" + str(RMSE))
    def execute(self, train_data, test_data):
        self.starttime = str(datetime.datetime.now())
        outfile("../log/NeuMF_dire.log", "\n\n")
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            if (epoch) % self.T == 0:
                self.test(test_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

class NFM_dire_add_MLP():

    def __init__(self, sess, num_user, num_item, learning_rate = 0.01, reg_rate = 0.01, epoch = 500, batch_size = 128, show_time = True, T =1, display_step= 1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        self.dire_num = 47
        print("NFM.")
        self.init_value = 0.1

    def build_network(self, feature_M, num_factor = 64, num_hidden = 128,added_factor = 4):
        init_value = self.init_value
        # model dependent arguments
        self.train_features = tf.placeholder(tf.int32, shape=[None, None])

        ############################
        self.dire_pos_indices = tf.placeholder(tf.int32, shape=[None,1])
        self.dire_neg_indices = tf.placeholder(tf.int32, shape=[None,1])
        emb_dire_pos = tf.Variable(tf.truncated_normal([self.dire_num, added_factor/2],
                                                       stddev=0.1 / math.sqrt(float(added_factor)), mean=0),
                                   name='emb_dire_pos', dtype=tf.float32)
        emb_dire_neg = tf.Variable(tf.truncated_normal([self.dire_num, added_factor/2],
                                                       stddev=init_value / math.sqrt(float(added_factor)), mean=0),
                                   name='emb_dire_neg', dtype=tf.float32)

        self.pos_dire_feature = tf.nn.embedding_lookup(emb_dire_pos, self.dire_pos_indices, name='dire_pos_feature')
        self.neg_dire_feature = tf.nn.embedding_lookup(emb_dire_neg, self.dire_neg_indices, name='dire_neg_feature')
        #############################
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.dropout_keep = tf.placeholder(tf.float32)

        self.feature_embeddings = tf.Variable(tf.random_normal([feature_M, num_factor], mean=0.0, stddev=0.01))

        self.feature_bias = tf.Variable(tf.random_uniform([feature_M, 1], 0.0, 0.0))
        self.bias = tf.Variable(tf.constant(0.0))
        self.pred_weight = tf.Variable(np.random.normal(loc=0, scale= np.sqrt(2.0 / (num_factor + num_hidden + added_factor)), size=(num_hidden, 1)),
                                                dtype=np.float32)

        nonzero_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, self.train_features)

        self.summed_features_embedding = tf.reduce_sum(nonzero_embeddings, 1)
        self.squared_summed_features_embedding = tf.square(self.summed_features_embedding)
        self.squared_features_embedding = tf.square(nonzero_embeddings)
        self.summed_squared_features_embedding = tf.reduce_sum(self.squared_features_embedding, 1)

        self.FM = 0.5 * tf.subtract( self.squared_summed_features_embedding, self.summed_squared_features_embedding)
        self.pos_dire_feature = tf.reduce_sum(self.pos_dire_feature, 1)
        self.neg_dire_feature = tf.reduce_sum(self.neg_dire_feature, 1)
        self.FM = tf.concat([self.FM, self.pos_dire_feature,self.neg_dire_feature], axis=1)
        # if batch_norm:
        #     self.FM = self
        layer_1 = tf.layers.dense(inputs=self.FM, units=num_hidden,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.FM =  tf.matmul(tf.nn.dropout(layer_1, self.dropout_keep), self.pred_weight)



        bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)
        self.f_b = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_bias, self.train_features), 1)
        b = self.bias * tf.ones_like(self.y)
        self.pred_rating = tf.add_n([bilinear, self.f_b, b])

        self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.pred_rating)) \
                    + tf.contrib.layers.l2_regularizer(self.reg_rate)(self.feature_embeddings)


        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):


        print("data preparation finished.")
        return self


    def train(self, train_data,add_train_feature):
        self.num_training = len(train_data['Y'])
        total_batch = int( self.num_training/ self.batch_size)

        rng_state = np.random.get_state()
        np.random.shuffle(train_data['Y'])
        np.random.set_state(rng_state)
        np.random.shuffle(train_data['X'])
        np.random.set_state(rng_state)
        np.random.shuffle(add_train_feature['X1'])
        np.random.set_state(rng_state)
        np.random.shuffle(add_train_feature['X2'])
        np.random.set_state(rng_state)
        np.random.shuffle(add_train_feature['Y'])
        # train
        loss_list = []
        start_time = time.time()
        for i in range(total_batch):
            batch_y = train_data['Y'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_x = train_data['X'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_add_x1 = add_train_feature['X1'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_add_x2 = add_train_feature['X2'][i * self.batch_size:(i + 1) * self.batch_size]
            loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict={self.train_features: batch_x,
                                                                              self.dire_pos_indices:batch_add_x1,
                                                                              self.dire_neg_indices: batch_add_x2,
                                                                              self.y: batch_y,
                                                                              self.dropout_keep:0.8})
            loss_list.append(loss)
        print("cost= %.9f" % (np.mean(np.array(loss_list))))
        if self.show_time:
            print(" %s seconds." % (time.time() - start_time))

    def test(self, test_data,add_test_feature):
        # error = 0
        # error_mae = 0
        # test_set = list(test_data.keys())
        # for (u, i) in test_set:
        #     pred_rating_test = self.predict([u], [i])
        #     error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
        #     error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        num_example = len(test_data['Y'])
        feed_dict = {self.train_features: test_data['X'], self.dire_pos_indices:add_test_feature['X1'],  self.dire_neg_indices: add_test_feature['X2'],self.y: test_data['Y'],self.dropout_keep: 1.0}
        predictions = self.sess.run((self.pred_rating), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(test_data['Y'], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

        print("RMSE:" + str(RMSE))
        outfile("../log/NeuMF_addMLP_dire.log", self.starttime + "\t" + "RMSE\t" + str(RMSE))
    def execute(self, train_data, test_data,add_train_feature,add_test_feature):
        self.starttime = str(datetime.datetime.now())
        outfile("../log/NeuMF_addMLP_dire.log", "\n\n")
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            print("Epoch: %04d;" % (epoch))
            self.train(train_data,add_train_feature)
            if (epoch) % self.T == 0:
                self.test(test_data,add_test_feature)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

class NFM_dire_add_MLP2():

    def __init__(self, sess, num_user, num_item, learning_rate = 0.01, reg_rate = 0.01, epoch = 500, batch_size = 128, show_time = True, T =1, display_step= 1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("NFM.")


    def build_network(self, feature_M, num_factor = 64, num_hidden = 128,added_factor = 2):
        # model dependent arguments
        self.train_features = tf.placeholder(tf.int32, shape=[None, None])
        self.train_features_add = tf.placeholder(tf.int32, shape=[None, 2])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.dropout_keep = tf.placeholder(tf.float32)

        self.feature_embeddings = tf.Variable(tf.random_normal([feature_M, num_factor], mean=0.0, stddev=0.01))

        self.feature_bias = tf.Variable(tf.random_uniform([feature_M, 1], 0.0, 0.0))
        self.bias = tf.Variable(tf.constant(0.0))
        self.pred_weight = tf.Variable(np.random.normal(loc=0, scale= np.sqrt(2.0 / (num_factor + num_hidden + added_factor)), size=(num_hidden, 1)),
                                                dtype=np.float32)

        nonzero_embeddings = tf.nn.embedding_lookup(self.feature_embeddings, self.train_features)

        self.summed_features_embedding = tf.reduce_sum(nonzero_embeddings, 1)
        self.squared_summed_features_embedding = tf.square(self.summed_features_embedding)
        self.squared_features_embedding = tf.square(nonzero_embeddings)
        self.summed_squared_features_embedding = tf.reduce_sum(self.squared_features_embedding, 1)

        self.FM = 0.5 * tf.subtract( self.squared_summed_features_embedding, self.summed_squared_features_embedding)
        self.train_features_add = tf.to_float(self.train_features_add)
        self.FM = tf.concat([self.FM, self.train_features_add], axis=1)
        # if batch_norm:
        #     self.FM = self
        layer_1 = tf.layers.dense(inputs=self.FM, units=num_hidden,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.FM =  tf.matmul(tf.nn.dropout(layer_1, self.dropout_keep), self.pred_weight)



        bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)
        self.f_b = tf.reduce_sum(tf.nn.embedding_lookup(self.feature_bias, self.train_features), 1)
        b = self.bias * tf.ones_like(self.y)
        self.pred_rating = tf.add_n([bilinear, self.f_b, b])

        self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.pred_rating)) \
                    + tf.contrib.layers.l2_regularizer(self.reg_rate)(self.feature_embeddings)


        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def prepare_data(self, train_data, test_data):


        print("data preparation finished.")
        return self


    def train(self, train_data,add_train_feature):
        self.num_training = len(train_data['Y'])
        total_batch = int( self.num_training/ self.batch_size)

        rng_state = np.random.get_state()
        np.random.shuffle(train_data['Y'])
        np.random.set_state(rng_state)
        np.random.shuffle(train_data['X'])
        np.random.set_state(rng_state)
        np.random.shuffle(add_train_feature['X'])
        np.random.set_state(rng_state)
        np.random.shuffle(add_train_feature['Y'])
        # train
        loss_list = []
        start_time = time.time()
        for i in range(total_batch):
            batch_y = train_data['Y'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_x = train_data['X'][i * self.batch_size:(i + 1) * self.batch_size]
            batch_add_x = add_train_feature['X'][i * self.batch_size:(i + 1) * self.batch_size]
            loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict={self.train_features: batch_x,
                                                                              self.train_features_add:batch_add_x,
                                                                              self.y: batch_y,
                                                                              self.dropout_keep:0.8})
            loss_list.append(loss)
        print("cost= %.9f" % (np.mean(np.array(loss_list))))
        if self.show_time:
            print(" %s seconds." % (time.time() - start_time))

    def test(self, test_data,add_test_feature):
        # error = 0
        # error_mae = 0
        # test_set = list(test_data.keys())
        # for (u, i) in test_set:
        #     pred_rating_test = self.predict([u], [i])
        #     error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
        #     error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        num_example = len(test_data['Y'])
        feed_dict = {self.train_features: test_data['X'], self.train_features_add:add_test_feature['X'] ,self.y: test_data['Y'],self.dropout_keep: 1.0}
        predictions = self.sess.run((self.pred_rating), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(test_data['Y'], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded,
                                         np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

        print("RMSE:" + str(RMSE))
        outfile("../log/NeuMF_addMLP_dire.log", self.starttime + "\t" + "RMSE\t" + str(RMSE))
    def execute(self, train_data, test_data,add_train_feature,add_test_feature):
        self.starttime = str(datetime.datetime.now())
        outfile("../log/NeuMF_addMLP_dire.log", "\n\n")
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            print("Epoch: %04d;" % (epoch))
            self.train(train_data,add_train_feature)
            if (epoch) % self.T == 0:
                self.test(test_data,add_test_feature)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)