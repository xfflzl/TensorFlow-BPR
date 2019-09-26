import heapq
import numpy as np

_model = None
_testRatings = None
_K = None

def evaluate_model(model, testRatings, K):
    global _model
    global _testRatings
    global _K
    _model = model
    _testRatings = testRatings
    _K = K
    num_rating = len(testRatings)

    res = list(map(eval_one_rating, range(num_rating)))

    hits = [r[0] for r in res]
    ndcgs = [r[1] for r in res]

    return hits, ndcgs

def eval_one_rating(idx):
    rating = _testRatings[idx]
    hr = ndcg = 0.0
    u, gtItem = rating[0], rating[1]
    map_item_score = {}
    # calculate the score of the ground truth item
    maxScore = _model.score(u, gtItem)

    # early stopping if there are K items larger than maxScore
    countLarger = 0
    early_stop = False
    for i in range(1, _model.num_item + 1):
        score_ui = _model.score(u, i)
        map_item_score[i] = score_ui

        if score_ui > maxScore:
            countLarger += 1
        if countLarger > _K:
            early_stop = True
            break
    if early_stop == False:
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)

    return hr, ndcg

def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    else:
        return 0

def getNDCG(ranklist, gtItem):
    for i in range(_K):
        item = ranklist[i]
        if item == gtItem:
            return np.log(2) / np.log(i+2)
    return 0
