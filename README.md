# TensorFlow-BPR

* This is a repo of implementation of Bayesian Personalized Ranking(BPR) and Matrix Factorization(MF) using TensorFlow APIs.

* A similar implementation can be found [here](https://github.com/hexiangnan/theano-BPR), which is written in Theano.

## Dataset

* [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) is used in this repo.

* The *u.data* file has 100000 ratings by 943 users on 1682 items. Each user has rated at least 20 movies. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of user id | item id | rating | timestamp. The time stamps are unix seconds since 1/1/1970 UTC.
* Note that the ratings are out-of-order.

## Requirement

* Python >= 3.7.0

* TensorFlow >= 1.14.0

## Contact

Email: xfflzl@mail.ustc.edu.cn