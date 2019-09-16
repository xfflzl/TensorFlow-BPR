import pandas as pd
from collections import defaultdict

def LoadRatingFile_LeaveKOut(datapath, splitter, K):
    '''
    The u.data file is a tab separated list of user id | item id | rating | timestamp.
    Each element of train is a list of [item, timestamp] pairs of one user, sorted by timestamp.
    Each element of test is a [user, item, timestamp] triple, sorted by timestamp.
    '''
    ui_data = pd.read_csv(datapath, sep=splitter, header=None, names=['uid', 'iid', 'rating', 'timestamp'], engine='python')

    train = defaultdict(list)
    test = []

    # load ratings into train
    item_pool = set() # aim to calculate #item
    num_ratings = ui_data.shape[0]
    for k in range(num_ratings):
        user, item, timestamp = ui_data.iat[k, 0], ui_data.iat[k, 1], ui_data.iat[k, 3]
        item_pool.add(item)
        train[user].append([item, timestamp])
    num_user = len(train)
    num_item = len(item_pool)

    for u in range(num_user):
        sorted(train[u], key=lambda x: x[-1]) # sorted by timestamp

    # split train / test set by leave-K-out protocol
    for u in range(1, num_user + 1): # Note that are numbered consecutively from 1.
        for _ in range(K):
            if len(train[u]) == 0: # no available interaction for test
                break
            else:
                test.append([u, train[u][-1][0], train[u][-1][1]]) # split the last interaction into testing set
                del train[u][-1] 
    
    test = sorted(test, key=lambda x: x[-1]) # sorted by timestamp

    return num_ratings, num_user, num_item, train, test