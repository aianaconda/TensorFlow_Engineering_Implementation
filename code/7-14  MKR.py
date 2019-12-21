import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.layers import base

class CrossCompressUnit(base.Layer):
    def __init__(self, dim, name=None):
       super(CrossCompressUnit, self).__init__(name)
       self.dim = dim
       self.f_vv = tf.layers.Dense(1, use_bias = False)
       self.f_ev = tf.layers.Dense(1, use_bias = False)
       self.f_ve = tf.layers.Dense(1, use_bias = False)
       self.f_ee = tf.layers.Dense(1, use_bias = False)
       self.bias_v = self.add_weight(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
       self.bias_e = self.add_weight(name='bias_e', shape=dim, initializer=tf.zeros_initializer())

    def __call__(self, inputs):
        v, e = inputs#v和e的形状为[batch_size, dim]
        v = tf.expand_dims(v, dim=2)#v的形状为 [batch_size, dim, 1]
        e = tf.expand_dims(e, dim=1)#e的形状为 [batch_size, 1, dim]

        c_matrix = tf.matmul(v, e)#c_matrix的形状为 [batch_size, dim, dim]
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        c_matrix = tf.reshape(c_matrix, [-1, self.dim])#c_matrix的形状为[batch_size * dim, dim]
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])

        v_output = tf.reshape( self.f_vv(c_matrix) + self.f_ev(c_matrix_transpose),
                              [-1, self.dim]) + self.bias_v#v_output的形状为[batch_size, dim]
        e_output = tf.reshape(self.f_ve(c_matrix) + self.f_ee(c_matrix_transpose),
                              [-1, self.dim]) + self.bias_e

        return v_output, e_output

class MKR(object):
    def __init__(self, args, n_users, n_items, n_entities, n_relations):
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_inputs()
        self._build_low_layers(args)
        self._build_high_layers(args)
        self._build_loss(args)
        self._build_train(args)

    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

        #收集训练参数，用于计算 l2 loss
        self.vars_rs = []
        self.vars_kge = []

    def _build_inputs(self):
        self.user_indices = tf.placeholder(tf.int32, [None], 'user_indices')
        self.item_indices = tf.placeholder(tf.int32, [None], 'item_indices')
        self.labels = tf.placeholder(tf.float32, [None], 'labels')
        self.head_indices = tf.placeholder(tf.int32, [None], 'head_indices')
        self.tail_indices = tf.placeholder(tf.int32, [None], 'tail_indices')
        self.relation_indices = tf.placeholder(tf.int32, [None], 'relation_indices')
        

    def _build_low_layers(self, args):
        #生成词嵌入
        self.user_emb_matrix = tf.get_variable('user_emb_matrix', [self.n_user, args.dim])
        self.item_emb_matrix = tf.get_variable('item_emb_matrix', [self.n_item, args.dim])
        self.entity_emb_matrix = tf.get_variable('entity_emb_matrix', [self.n_entity, args.dim])
        self.relation_emb_matrix = tf.get_variable('relation_emb_matrix', [self.n_relation, args.dim])

        #获取指定输入对应的词嵌入，形状为[batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.item_embeddings = tf.nn.embedding_lookup(self.item_emb_matrix, self.item_indices)
        self.head_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.head_indices)
        self.relation_embeddings = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relation_indices)
        self.tail_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.tail_indices)

        for _ in range(args.L):#按指定参数，构建多层MKR结构
            #定义全连接
            user_mlp = tf.layers.Dense(args.dim, activation=tf.nn.relu)
            tail_mlp = tf.layers.Dense(args.dim, activation=tf.nn.relu)
            cc_unit = CrossCompressUnit(args.dim)#定义CrossCompress单元
            #实现MKR结构的正向处理
            self.user_embeddings = user_mlp(self.user_embeddings)
            self.tail_embeddings = tail_mlp(self.tail_embeddings)
            self.item_embeddings, self.head_embeddings = cc_unit([self.item_embeddings, self.head_embeddings])
            #收集训练参数
            self.vars_rs.extend(user_mlp.variables)
            self.vars_kge.extend(tail_mlp.variables) 
            self.vars_rs.extend(cc_unit.variables)
            self.vars_kge.extend(cc_unit.variables)
            
            
    def _build_high_layers(self, args):
        #RS模型的处理
        use_inner_product = True #指定相似度分数计算的方式
        if use_inner_product:    #内积方式
            #self.scores的形状为[batch_size]
            self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        else:
            #self.user_item_concat的形状为[batch_size, dim * 2]
            self.user_item_concat = tf.concat([self.user_embeddings, self.item_embeddings], axis=1)
            for _ in range(args.H - 1):
                rs_mlp = tf.layers.Dense(args.dim * 2, activation=tf.nn.relu)
                #self.user_item_concat的形状为[batch_size, dim * 2]
                self.user_item_concat = rs_mlp(self.user_item_concat)
                self.vars_rs.extend(rs_mlp.variables)
            #定义全连接
            rs_pred_mlp = tf.layers.Dense(1, activation=tf.nn.relu)
            #self.scores的形状为[batch_size]
            self.scores = tf.squeeze(rs_pred_mlp(self.user_item_concat))
            self.vars_rs.extend(rs_pred_mlp.variables)  #收集参数
        self.scores_normalized = tf.nn.sigmoid(self.scores)

        #KGE模型的处理
        #self.head_relation_concat的形状为[batch_size, dim * 2]
        self.head_relation_concat = tf.concat([self.head_embeddings, self.relation_embeddings], axis=1)
        for _ in range(args.H - 1):
            kge_mlp = tf.layers.Dense(args.dim * 2, activation=tf.nn.relu)
            self.head_relation_concat = kge_mlp(self.head_relation_concat)
            self.vars_kge.extend(kge_mlp.variables)

        kge_pred_mlp = tf.layers.Dense(args.dim, activation=tf.nn.relu)
        self.tail_pred = kge_pred_mlp(self.head_relation_concat)
        self.vars_kge.extend(kge_pred_mlp.variables)
        self.tail_pred = tf.nn.sigmoid(self.tail_pred)

        self.scores_kge = tf.nn.sigmoid(tf.reduce_sum(self.tail_embeddings * self.tail_pred, axis=1))
        self.rmse = tf.reduce_mean(
            tf.sqrt(tf.reduce_sum(tf.square(self.tail_embeddings - self.tail_pred), axis=1) / args.dim))

    def _build_loss(self, args):
        #计算RS模型的loss值
        self.base_loss_rs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))
        self.l2_loss_rs = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings)
        for var in self.vars_rs:
            self.l2_loss_rs += tf.nn.l2_loss(var)
        self.loss_rs = self.base_loss_rs + self.l2_loss_rs * args.l2_weight

        #计算KGE模型的loss值
        self.base_loss_kge = -self.scores_kge
        self.l2_loss_kge = tf.nn.l2_loss(self.head_embeddings) + tf.nn.l2_loss(self.tail_embeddings)
        for var in self.vars_kge: #计算L2正则
            self.l2_loss_kge += tf.nn.l2_loss(var)
        self.loss_kge = self.base_loss_kge + self.l2_loss_kge * args.l2_weight

    def _build_train(self, args):
        self.optimizer_rs = tf.train.AdamOptimizer(args.lr_rs).minimize(self.loss_rs)
        self.optimizer_kge = tf.train.AdamOptimizer(args.lr_kge).minimize(self.loss_kge)

    def train_rs(self, sess, feed_dict):
        return sess.run([self.optimizer_rs, self.loss_rs], feed_dict)

    def train_kge(self, sess, feed_dict):
        return sess.run([self.optimizer_kge, self.rmse], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
