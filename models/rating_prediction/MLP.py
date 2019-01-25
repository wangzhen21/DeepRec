'''
Created on Jan 23, 2018

@author: v-lianji
'''
from __future__ import print_function
import math
import tensorflow as tf
import time
import numpy as np
import datetime
from utils.load_data.load_data_rating import *


def RMSE(error, num):
    return np.sqrt(error / num)


def MAE(error_mae, num):
    return (error_mae / num)

class MLP():
    def __init__(self, args, num_users, num_items,batch_size=128,
                 show_time=False, T=2, display_step=1000):
        self.num_users, self.num_items = num_users, num_items
        self.lr = args.lr
        self.learner = args.learner
        self.init_stddev = args.init_stddev
        self.loss = args.loss
        self.lambda_id_emb = args.reg_id_embedding
        self.lambda_others = args.reg_others
        self.eta = args.eta
        self.layers = eval(args.layers)
        self.lambda_layers = eval(args.reg_layers)
        self.epochs = args.epochs
        self.T = 1
        # self.build_model()
        self.batch_size = batch_size
        self.show_time = show_time
        self.display_step = display_step
        print("MF.")

    def build_core_model(self, user_indices, item_indices):

        init_value = self.init_stddev

        emb_user = tf.Variable(tf.truncated_normal([self.num_users, self.layers[0] // 2],
                                                   stddev=init_value / math.sqrt(float(self.layers[0] // 2)), mean=0),
                               name='user_embedding', dtype=tf.float32)
        emb_item = tf.Variable(tf.truncated_normal([self.num_items, self.layers[0] // 2],
                                                   stddev=init_value / math.sqrt(float(self.layers[0] // 2)), mean=0),
                               name='item_embedding', dtype=tf.float32)
        user_feature = tf.nn.embedding_lookup(emb_user, user_indices, name='user_feature')
        item_feature = tf.nn.embedding_lookup(emb_item, item_indices, name='item_feature')

        hidden_layers = [tf.concat([user_feature, item_feature], 1)]

        model_params = [emb_user, emb_item]

        for i in range(1, len(self.layers)):
            w_hidden_layer = tf.Variable(
                tf.truncated_normal([self.layers[i - 1], self.layers[i]], stddev=init_value, mean=0),
                name='w_hidden_' + str(i), dtype=tf.float32)
            b_hidden_layer = tf.Variable(tf.truncated_normal([self.layers[i]], stddev=init_value * 0.1, mean=0),
                                         name='b_hidden_' + str(i), dtype=tf.float32)
            cur_layer = tf.nn.xw_plus_b(hidden_layers[-1], w_hidden_layer, b_hidden_layer)
            cur_layer = tf.nn.relu(cur_layer)
            hidden_layers.append(cur_layer)
            model_params.append(w_hidden_layer)
            model_params.append(b_hidden_layer)

        return hidden_layers[-1], self.layers[-1], model_params

    def build_network(self, user_indices=None, item_indices=None):

        if not user_indices:
            user_indices = tf.placeholder(tf.int32, [None])
        self.user_indices = user_indices
        if not item_indices:
            item_indices = tf.placeholder(tf.int32, [None])
        self.item_indices = item_indices

        self.ratings = tf.placeholder(tf.float32, [None])

        model_vector, model_len, model_params = self.build_core_model(user_indices, item_indices)

        self.output, self.loss, self.error, self.raw_error, self.train_step = self.build_train_model(model_vector,
                                                                                                     model_len,
                                                                                                     self.ratings,
                                                                                                     model_params)

    def build_train_model(self, model_vector, model_len, ratings, model_params):
        init_value = self.init_stddev

        w_output = tf.Variable(tf.truncated_normal([model_len, 1], stddev=init_value, mean=0), name='w_output',
                               dtype=tf.float32)
        b_output = tf.Variable(tf.truncated_normal([1], stddev=init_value * 0.01, mean=0), name='b_output',
                               dtype=tf.float32)
        model_params.append(w_output)
        model_params.append(b_output)
        raw_predictions = tf.nn.xw_plus_b(model_vector, w_output, b_output, name='output')

        output = tf.reshape(tf.sigmoid(raw_predictions) * 5, [-1])

        with tf.name_scope('error'):
            type_of_loss = self.loss
            if type_of_loss == 'cross_entropy_loss':
                raw_error = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(raw_predictions, [-1]),
                                                                    labels=tf.reshape(self.ratings, [-1]))
                error = tf.reduce_mean(
                    raw_error,
                    name='error/cross_entropy_loss'
                )
            elif type_of_loss == 'square_loss' or type_of_loss == 'rmse':
                raw_error = tf.squared_difference(output, ratings, name='error/squared_diff')
                error = tf.reduce_mean(raw_error, name='error/mean_squared_diff')
            elif type_of_loss == 'log_loss':
                raw_error = tf.losses.log_loss(predictions=output, labels=ratings)
                error = tf.reduce_mean(raw_error, name='error/mean_log_loss')

            l2_norm = 0
            for par in model_params:
                l2_norm += tf.nn.l2_loss(par) * self.lambda_others
            r'''
            l2_norm += tf.nn.l2_loss(emb_user) * self.lambda_id_emb
            l2_norm += tf.nn.l2_loss(emb_item) * self.lambda_id_emb
            l2_norm += tf.nn.l2_loss(w_output) * self.lambda_others
            l2_norm += tf.nn.l2_loss(b_output) * self.lambda_others
            '''

            loss = error + l2_norm

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  ##--
        with tf.control_dependencies(update_ops):
            type_of_opt = self.learner
            if type_of_opt == 'adadelta':
                train_step = tf.train.AdadeltaOptimizer(self.eta).minimize(loss, var_list=model_params)  #
            elif type_of_opt == 'sgd':
                train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss, var_list=model_params)
            elif type_of_opt == 'adam':
                train_step = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=model_params)
            elif type_of_opt == 'ftrl':
                train_step = tf.train.FtrlOptimizer(self.lr).minimize(loss, var_list=model_params)
            else:
                train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss, var_list=model_params)

        return output, loss, error, raw_error, train_step

    def train(self, train_data):
        self.num_training = len(train_data[0])
        total_batch = int( self.num_training/self.batch_size)

        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(np.array(train_data[0])[idxs])
        item_random = list(np.array(train_data[1])[idxs])
        rating_random = list(np.array(train_data[2])[idxs])

        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        user_random = list(np.array(user_random)[idxs])
        item_random = list(np.array(item_random)[idxs])
        rating_random = list(np.array(rating_random)[idxs])
        # train
        loss_per_epoch, error_per_epoch = 0, 0
        for i in range(total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]
            _, loss, error = self.sess.run([self.train_step, self.loss, self.raw_error], { self.user_indices : batch_user, self.item_indices : batch_item, self.ratings : batch_rating})
            loss_per_epoch += loss
            error_per_epoch += error
        error_per_epoch /= total_batch
        loss_per_epoch /= total_batch
        t2 = time.time()
        print('loss= %.4f\t[%.1f s]' % (loss_per_epoch, time.time() - t2))
        if i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)) + "\n")
                if self.show_time:
                    print("one iteration: %s seconds." % (time.time() - start_time) + "\n")

    def test(self, test_data):
        pred_rating_test = self.predict(test_data[0], test_data[1])
        error = np.sum(np.power((np.array(test_data[2]) - np.array(pred_rating_test)),2))
        error_mae = np.sum(np.abs(np.array(test_data[2]) - np.array(pred_rating_test)))
        out_rmse = str(RMSE(error, len(test_data[0])))
        out_mae= str(MAE(error_mae, len(test_data[0])))
        print("RMSE:" + out_rmse + "; MAE:" + out_mae)
        outfile("../log/MLP.log",self.starttime  + "\t" + "RMSE:" + out_rmse + "; MAE:" + out_mae)

    def execute(self, train_data, test_data):
        self.starttime = str(datetime.datetime.now())
        outfile("../log/MLP.log", "\n\n")
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        t1 = time.time()
        self.test(test_data)
        for epoch in range(self.epochs):
            print("Epoch: %04d;" % (epoch))
            if (epoch) % self.T == 0:
                self.test(test_data)
            self.train(train_data)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.sess.run([self.output], feed_dict={self.user_indices : user_id, self.item_indices : item_id})