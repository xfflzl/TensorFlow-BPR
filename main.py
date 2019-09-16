import tensorflow as tf
from dataloader import LoadRatingFile_LeaveKOut
from MFbpr import MFbpr

if __name__ == '__main__':
    datapath = './data/u.data'
    splitter = '\t'
    leave_k_out = 1

    num_ratings, num_user, num_item, train, test = LoadRatingFile_LeaveKOut(datapath, splitter, leave_k_out)
    print('Successfully load data from {}.'.format(datapath))
    print('#users: {}, #items: {}, #ratings: {}'.format(num_user, num_item, num_ratings))

    hidden_dims = 10
    learning_rate = 0.01
    reg = 0.01
    init_mean = 0
    init_stdev = 0.01
    maxIter = 5
    batch_size = 32
    topK = 100
    Optimizer = tf.train.AdamOptimizer

    # Run model
    bpr = MFbpr(train, test, num_user, num_item, 
                hidden_dims, Optimizer, learning_rate, reg, topK, init_mean, init_stdev)
    bpr.build_model(maxIter, batch_size)