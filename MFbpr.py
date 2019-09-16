import time
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from evaluate import evaluate_model

class MFbpr(object):
    '''
    Implementation of MF + BPR using tensorflow APIs
    '''
    def __init__(self, train, test, num_user, num_item, hidden_dims=20, 
                 Optimizer=tf.train.GradientDescentOptimizer, learning_rate=0.01, 
                 reg=0.0001, topK=100, init_mean=0.0, init_stdev=0.1):
        self.train = train
        self.test = test
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_dims = hidden_dims
        self.Optimizer = Optimizer
        self.learning_rate = learning_rate
        self.reg = reg
        self.topK = topK

        self.items_of_user = defaultdict(set)
        self.num_rating = 0
        for u in range(1, self.num_user + 1):
            for inta in train[u]: # interaction
                self.items_of_user[u].add(inta[0])
                self.num_rating += 1

        self.u = tf.placeholder(tf.int32, [None])
        self.i = tf.placeholder(tf.int32, [None])
        self.j = tf.placeholder(tf.int32, [None])

        # latent matrices of users and items
        '''
        Since users and items are numbered consecutively from 1, 
        the first dimension of latent matrices is set to (num + 1), 
        and the first row doesn't do anything.
        '''
        self.user_emb_w = tf.get_variable('user_emb_w', [self.num_user + 1, self.hidden_dims], 
                                          initializer=tf.random_normal_initializer(init_mean, init_stdev))
        self.item_emb_w = tf.get_variable('item_emb_w', [self.num_item + 1, self.hidden_dims], 
                                          initializer=tf.random_normal_initializer(init_mean, init_stdev))

        self.u_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.item_emb_w, self.j)

        # calculate loss of the sample
        y_ui = tf.reduce_sum(tf.multiply(self.u_emb, self.i_emb), axis=1, keep_dims=True)
        y_uj = tf.reduce_sum(tf.multiply(self.u_emb, self.j_emb), axis=1, keep_dims=True)
        l2_reg = self.reg * tf.add_n([tf.reduce_sum(tf.multiply(self.u_emb, self.u_emb)), 
                                 tf.reduce_sum(tf.multiply(self.i_emb, self.i_emb)), 
                                 tf.reduce_sum(tf.multiply(self.j_emb, self.j_emb))])
        bprloss = l2_reg - tf.reduce_mean(tf.log(tf.sigmoid(y_ui - y_uj)))

        # optimization
        self.sgd_step = self.Optimizer(self.learning_rate).minimize(bprloss)

    def build_model(self, maxIter=100, batch_size=32):
        self.maxIter = maxIter
        self.batch_size = batch_size
        print('Training MF-BPR model with: learning_rate={}, reg={}, hidden_dims={}, #epoch={}, batch_size={}.'.format(
              self.learning_rate, self.reg, self.hidden_dims, self.maxIter, self.batch_size))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # training process
            t1 = time.time()
            for iteration in range(maxIter):
                # each training epoch
                for _ in range(self.num_rating // self.batch_size):
                    uij_train = self.get_batch()
                    sess.run([self.sgd_step], feed_dict={ # optimization
                             self.u: uij_train[:, 0], 
                             self.i: uij_train[:, 1], 
                             self.j: uij_train[:, 2]})
                print('Have finished epoch {}.'.format(iteration + 1))

            # check performance
            t2 = time.time()
            variable_names = [v.name for v in tf.trainable_variables()]
            self.parameters = sess.run(variable_names)
            # self.parameters[0] ==> latent matrix for users
            # self.parameters[1] ==> latent matrix for items
            hits, ndcgs = evaluate_model(self, self.test, self.topK)
            print('Iter: {} [{:.2f} s] HitRatio@{} = {:.4f}, NDCG@{} = {:.4f} [{:.2f} s]'.format(
                  iteration + 1, t2 - t1, self.topK, np.array(hits).mean(), self.topK, np.array(ndcgs).mean(), time.time() - t2))

    def score(self, u, i):
        return np.inner(self.parameters[0][u], self.parameters[1][i])

    def get_batch(self):
        t = []
        for _ in range(self.batch_size):
            # sample a user
            _u = random.sample(range(1, self.num_user + 1), 1)[0]
            # sample a positive item
            _i = random.sample(self.items_of_user[_u], 1)[0]
            # sample a negative item
            _j = random.sample(range(1, self.num_item + 1), 1)[0]
            while _j in self.items_of_user[_u]:
                _j = random.sample(range(1, self.num_item + 1), 1)[0]
            t.append([_u, _i, _j])
        return np.asarray(t)
