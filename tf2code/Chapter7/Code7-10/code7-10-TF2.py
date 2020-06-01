# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
"""


import os

DATASET_PATH= 'ml-latest-small'


RATINGS_CSV = os.path.join(DATASET_PATH, 'ratings.csv')

#with open(RATINGS_CSV, 'r') as f:
#    for _ in range(10):
#        print(f.readline().strip())

import collections
import csv

Rating = collections.namedtuple('Rating', ['user_id', 'item_id', 'rating', 'timestamp'])
ratings = list()
with open(RATINGS_CSV, newline='') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for user_id, item_id, rating, timestamp in reader:
        ratings.append(Rating(user_id, item_id, float(rating), int(timestamp)))

ratings = sorted(ratings, key=lambda r: r.timestamp)
print('Ratings: {:,}'.format(len(ratings)))

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()
users_from_idx = sorted(set(r.user_id for r in ratings), key=int)#用户ID
users_from_idx = dict(enumerate(users_from_idx))
users_to_idx = dict((user_id, idx) for idx, user_id in users_from_idx.items())
print('User Index:',[users_from_idx[i] for i in range(2)])


items_from_idx = sorted(set(r.item_id for r in ratings), key=int)#电影ID
items_from_idx = dict(enumerate(items_from_idx))
items_to_idx = dict((item_id, idx) for idx, item_id in items_from_idx.items())
print('Item Index:',[items_from_idx[i] for i in range(2)])




sess = tf.compat.v1.InteractiveSession()#将id 与电影交叉。评分填入
indices = [(users_to_idx[r.user_id], items_to_idx[r.item_id]) for r in ratings]
values = [r.rating for r in ratings]
n_rows = len(users_from_idx)
n_cols = len(items_from_idx)
shape = (n_rows, n_cols)

P = tf.SparseTensor(indices, values, shape)

print(P)
print('Total values: {:,}'.format(n_rows * n_cols))

from tensorflow.contrib.factorization import WALSModel

k = 10
n = 10
reg = 1e-1

model = WALSModel(
    n_rows,
    n_cols,
    k,
    regularization=reg,
    unobserved_weight=0)

row_factors = tf.nn.embedding_lookup(
    params=model.row_factors,
    ids=tf.range(model._input_rows),
    partition_strategy="div")
col_factors = tf.nn.embedding_lookup(
    params=model.col_factors,
    ids=tf.range(model._input_cols),
    partition_strategy="div")

row_indices, col_indices = tf.split(P.indices,
                                    axis=1,
                                    num_or_size_splits=2)
gathered_row_factors = tf.gather(row_factors, row_indices)
gathered_col_factors = tf.gather(col_factors, col_indices)
approx_vals = tf.squeeze(tf.matmul(gathered_row_factors,
                                   gathered_col_factors,
                                   adjoint_b=True))
P_approx = tf.SparseTensor(indices=P.indices,
                           values=approx_vals,
                           dense_shape=P.dense_shape)

E = tf.sparse.add(a=P, b=P_approx * (-1))
E2 = tf.square(E)
n_P = P.values.shape[0].value
rmse_op = tf.sqrt(tf.sparse.reduce_sum(E2) / n_P)

row_update_op = model.update_row_factors(sp_input=P)[1]
col_update_op = model.update_col_factors(sp_input=P)[1]

model.initialize_op.run()
model.worker_init.run()
for _ in range(n):
    # Update Users
    model.row_update_prep_gramian_op.run()
    model.initialize_row_update_op.run()
    row_update_op.run()
    # Update Items
    model.col_update_prep_gramian_op.run()
    model.initialize_col_update_op.run()
    col_update_op.run()

    print('RMSE: {:,.3f}'.format(rmse_op.eval()))

user_factors = model.row_factors[0].eval()
item_factors = model.col_factors[0].eval()

print('User factors shape:', user_factors.shape)
print('Item factors shape:', item_factors.shape)

c = collections.Counter(r.user_id for r in ratings)
user_id, n_ratings = c.most_common(1)[0]
print('评论最多的用户{}: {:,d}'.format(user_id, n_ratings))#userid为 547:评论了 2,391次。是最多的

r = next(r for r in reversed(ratings) if r.user_id == user_id and r.rating == 5.0)#找一条评论为5的数据
print('该用户最后一条5分记录 {}:\n'.format(user_id))
print(r)

#根据该用户和电影的索引在分解矩阵中取值，并计算模型的准确度
i = users_to_idx[r.user_id]
j = items_to_idx[r.item_id]

u = user_factors[i]
print('Factors for user {}:\n'.format(r.user_id))
print(u)
print()

v = item_factors[j]
print('Factors for item {}:\n'.format(r.item_id))
print(v)
print()

p = np.dot(u, v)
print('Approx. rating: {:,.3f}, diff={:,.3f}, {:,.3%}'.format(p, r.rating - p, p/r.rating))#预测结果

#推荐排名
V = item_factors
user_P = np.dot(V, u)
print('预测出用户所有的评分:', user_P.shape)

user_items = set(ur.item_id for ur in ratings if ur.user_id == user_id)#该用户评论的电影

user_ranking_idx = sorted(enumerate(user_P), key=lambda p: p[1], reverse=True)
user_ranking_raw = ((items_from_idx[j], p) for j, p in user_ranking_idx)
user_ranking = [(item_id, p) for item_id, p in user_ranking_raw if item_id not in user_items]#找到该用户没有评论过的所有电影评分

top10 = user_ranking[:10]#取出前10个

print('Top 10 items:\n')
for k, (item_id, p) in enumerate(top10):  #得到该用户喜欢电影的排名
    print('[{}] {} {:,.2f}'.format(k+1, item_id, p))








