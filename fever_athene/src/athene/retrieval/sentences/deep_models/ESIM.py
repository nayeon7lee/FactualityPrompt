import os
import random
import time
from math import ceil

import numpy as np
import tensorflow as tf
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

he_init = tf.contrib.layers.variance_scaling_initializer()
dim_fasttext = 300

tf.AUTO_REUSE = False

class ESIM:

    def __init__(self, optimizer=tf.train.AdamOptimizer, h_max_length=10, s_max_length=40, learning_rate=0.0001,
                 batch_size=128, activation=tf.nn.tanh, initializer=he_init, num_epoch=100, dropout_rate=None,
                 embedding=None, word_dict=None, max_check_without_progress=3, model_store_dir=None, random_state=None,
                 l2_lambda=0, trainable=False, share_rnn=False, num_units=128,namespace=""):

        self.optimizer = optimizer
        self.h_max_length = h_max_length
        self.s_max_length = s_max_length
        self.learning_rate = learning_rate
        self.embedding = embedding
        self.word_dict = word_dict
        self.batch_size = batch_size
        self.activation = activation
        self.num_epoch = num_epoch
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.max_checks_without_progress = max_check_without_progress
        self.random_state = random_state
        self.model_store_dir = model_store_dir if model_store_dir is not None else "model"
        self.l2_lambda = l2_lambda
        self.trainable = trainable
        self.share_rnn = share_rnn
        self.num_units = num_units
        self.namespace = namespace
        self._session = None
        self._graph = None
        print("NAMESPACE {}".format(namespace))
        # self.logger = LogHelper.get_logger(self.__class__.__name__)

    # use gru cell for rnn model
    def gru_cell(self, num_neuron):

        gru = tf.contrib.rnn.GRUCell(num_neuron)
        if self.dropout_rate:
            gru = tf.contrib.rnn.DropoutWrapper(gru, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        return gru

    # use lstm cell for rnn model
    def _lstm_cell(self, hidden_size):
        lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        if self.dropout_rate:
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        return lstm

    def _bidirectional_rnn(self, inputs, inputs_length, num_units, scope=None):
        """
        :param inputs:
        :param inputs_length:
        :param num_units:
        :param scope:
        :return:
        """

        with tf.variable_scope(scope or 'birnn'):
            rnn_cells = self._lstm_cell(num_units)
            ((fw_outputs, bw_outputs), (fw_states, bw_states)) = tf.nn.bidirectional_dynamic_rnn(rnn_cells, rnn_cells,
                                                                                                 inputs, inputs_length,
                                                                                                 dtype=tf.float32)
            outputs = tf.concat([fw_outputs, bw_outputs], axis=2)

        return outputs

    def _mask_3d(self, inputs, sentence_lengths, mask_value, dimension=2):

        if dimension == 1:
            inputs = tf.transpose(inputs, [0, 2, 1])

        time_steps1 = tf.shape(inputs)[1]
        time_steps2 = tf.shape(inputs)[2]

        pad_values = mask_value * tf.ones_like(inputs, dtype=tf.float32)
        mask = tf.sequence_mask(sentence_lengths, time_steps2)

        mask_3d = tf.tile(tf.expand_dims(mask, 1), (1, time_steps1, 1))
        masked = tf.where(mask_3d, inputs, pad_values)

        if dimension == 1:
            masked = tf.transpose(masked, [0, 2, 1])
        return masked

    def _inter_atten(self, claim, sent, claim_lengths, sent_lengths):

        with tf.variable_scope('inter-attention'):
            sent = tf.transpose(sent, [0, 2, 1])
            attention = tf.matmul(claim, sent)

            masked = self._mask_3d(attention, sent_lengths, -np.inf)
            att_sent1 = self._atten_softmax3d(masked)

            att_transpose = tf.transpose(attention, [0, 2, 1])
            masked = self._mask_3d(att_transpose, claim_lengths, -np.inf)
            att_sent2 = self._atten_softmax3d(masked)
            self.att_sent2 = att_sent2

            alpha = tf.matmul(att_sent2, claim, name="alpha")
            # self.alpha = alpha
            sent = tf.transpose(sent, [0, 2, 1])
            beta = tf.matmul(att_sent1, sent, name="beta")

        return alpha, beta

    def _atten_softmax3d(self, inputs):

        shape = tf.shape(inputs)
        num_units = shape[2]
        inputs = tf.reshape(inputs, tf.stack([-1, num_units]))
        soft_max = tf.nn.softmax(inputs)
        soft_max = tf.reshape(soft_max, shape)
        return soft_max

    def mlp(self, outputs):
        #with tf.variable_scope(self.namespace):
        outputs = tf.layers.dense(outputs, 256, activation=tf.nn.tanh, kernel_initializer=self.initializer, name="dense")

        if self.dropout_rate:
            outputs = tf.layers.dropout(outputs, rate=self.dropout_rate, training=self._training)

        return outputs

    def _construct_graph(self):

        if self.random_state:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        X_h = tf.placeholder(tf.int32, shape=[None, self.h_max_length], name="X_heads")
        X_s = tf.placeholder(tf.int32, shape=[None, self.s_max_length], name="X_sents")
        X_h_length = tf.placeholder(tf.int32, shape=[None, ], name="X_h_lengths")
        X_s_length = tf.placeholder(tf.int32, shape=[None, ], name="X_s_lengths")

        if self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=[], name="training")
            self.keep_prob = tf.cond(self._training, lambda: tf.constant(1 - self.dropout_rate),
                                     lambda: tf.constant(1.0))
        else:
            self._training = None

        try:
            with tf.variable_scope("embedding_lookup", reuse=True):
                embedding = tf.get_variable("embedding",
                                            initializer=self.embedding,
                                            dtype=tf.float32,
                                            trainable=self.trainable)
                embed_h = tf.nn.embedding_lookup(embedding, ids=X_h)
                embed_s = tf.nn.embedding_lookup(embedding, ids=X_s)
        except ValueError:
            with tf.variable_scope("embedding_lookup", reuse=False):
                embedding = tf.get_variable("embedding",
                                            initializer=self.embedding,
                                            dtype=tf.float32,
                                            trainable=self.trainable)
                embed_h = tf.nn.embedding_lookup(embedding, ids=X_h)
                embed_s = tf.nn.embedding_lookup(embedding, ids=X_s)

        if self.share_rnn:
            with tf.variable_scope("encode_rnn", reuse=tf.AUTO_REUSE):
                h_encodings = self._bidirectional_rnn(embed_h, X_h_length, self.num_units)
                s_encodings = self._bidirectional_rnn(embed_s, X_s_length, self.num_units)
        else:
            #with tf.variable_scope(self.namespace):
            h_encodings = self._bidirectional_rnn(embed_h, X_h_length, self.num_units, scope="h_encode_rnn")
            s_encodings = self._bidirectional_rnn(embed_s, X_s_length, self.num_units, scope="s_endode_rnn")

        sent_attends, claim_attends = self._inter_atten(h_encodings, s_encodings, X_h_length, X_s_length)

        claim_diff = tf.subtract(h_encodings, claim_attends)
        claim_mul = tf.multiply(h_encodings, claim_attends)

        sent_diff = tf.subtract(s_encodings, sent_attends)
        sent_mul = tf.multiply(s_encodings, sent_attends)

        m_claim = tf.concat([h_encodings, claim_attends, claim_diff, claim_mul], axis=2)
        m_sent = tf.concat([s_encodings, sent_attends, sent_diff, sent_mul], axis=2)

        if self.share_rnn:
            with tf.variable_scope("infer_rnn", reuse=tf.AUTO_REUSE):
                h_infer = self._bidirectional_rnn(m_claim, X_h_length, self.num_units)
                s_infer = self._bidirectional_rnn(m_sent, X_s_length, self.num_units)
        else:
            #with tf.variable_scope(self.namespace):
            h_infer = self._bidirectional_rnn(m_claim, X_h_length, self.num_units, scope="h_infer_rnn")
            s_infer = self._bidirectional_rnn(m_sent, X_s_length, self.num_units, scope="s_infer_rnn")

        claim_sum = tf.reduce_sum(h_infer, axis=1)
        claim_mask = tf.cast(tf.sequence_mask(X_h_length), tf.float32)
        claim_ave = tf.div(claim_sum, tf.reduce_sum(claim_mask, axis=1, keepdims=True))
        claim_max = tf.reduce_max(h_infer, axis=1)

        sent_sum = tf.reduce_sum(s_infer, axis=1)
        sent_mask = tf.cast(tf.sequence_mask(X_s_length), tf.float32)
        sent_ave = tf.div(sent_sum, tf.reduce_sum(sent_mask, axis=1, keepdims=True))
        sent_max = tf.reduce_max(s_infer, axis=1)

        v = tf.concat([claim_ave, claim_max, sent_ave, sent_max], axis=1)

        dense_output = self.mlp(v)

        #with tf.variable_scope(self.namespace):
        scores = tf.layers.dense(dense_output, 1, name="dense_1")

        pos = tf.strided_slice(scores, [0], [self.batch_size], [2])
        neg = tf.strided_slice(scores, [1], [self.batch_size], [2])
        loss = tf.reduce_mean(tf.maximum(0.0, 1.0 + neg - pos))

        optimizer = self.optimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        print("GET VARS")
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        # print("\n".join([str(el) for el in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))

        # if self.tensorboard_logdir:
        #     now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        #     tb_logdir = self.tensorboard_logdir + "/run{}".format(now)
        #     cost_summary = tf.summary.scalar("validation_loss", loss)
        #     merged_summary = tf.summary.merge_all()
        #     file_writer = tf.summary.FileWriter(
        #         tb_logdir, tf.get_default_graph())
        #
        #     self._merged_summary = merged_summary
        #     self._file_writer = file_writer

        self._X_h, self._X_s, self._X_h_length, self._X_s_length = X_h, X_s, X_h_length, X_s_length
        self.h_infer = h_infer
        self.h_encodings = h_encodings
        self.attend = claim_attends
        self.scores = scores
        self._init = init
        self._saver = saver
        self._loss = loss
        self._training_ops = training_op

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_parameters(self):

        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_parameters(self, model_parameters):

        gvar_names = list(model_parameters.keys())

        print(gvar_names)

        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}

        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_parameters[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def padding(self, sents, word_dict, max_length):
        _PAD_ = word_dict['[PAD]']
        lengths = []
        padded_sents = []
        for sent in sents:
            lengths.append(len(sent))
            if len(sent) < max_length:
                sent.extend([_PAD_] * (max_length - len(sent)))
                padded_sents.append(sent)
            else:
                padded_sents.append(sent)
        padded_sents = np.asarray(padded_sents, np.int32)
        lengths = np.asarray(lengths, np.int32)
        return padded_sents, lengths

    def generate_batch(self, X):

        batch_size = self.batch_size // 2
        for batch_i in range(0, ceil(len(X) / batch_size)):
            start_i = batch_i * batch_size
            end_i = start_i + batch_size
            if start_i == len(X):
                break
            if end_i > len(X):
                end_i = len(X)

            X_batch = X[start_i:end_i]
            X_claims_batch = []
            X_sents_batch = []
            for claim, pos, neg in X_batch:
                X_claims_batch.append(claim)
                X_claims_batch.append(claim)
                X_sents_batch.append(pos)
                X_sents_batch.append(neg)

            X_h_batch, X_h_lengths_batch = self.padding(X_claims_batch, self.word_dict, self.h_max_length)
            X_s_batch, X_s_lengths_batch = self.padding(X_sents_batch, self.word_dict, self.s_max_length)

            yield X_h_batch, X_s_batch, X_h_lengths_batch, X_s_lengths_batch

    def generate_dev_batch(self, dev):

        claims = [claim for claim, _ in dev]
        sents = [sent for _, sent in dev]
        for batch_i in range(0, ceil(len(claims) / 128)):
            start_i = batch_i * 128
            end_i = start_i + 128
            if start_i == len(claims):
                break
            if end_i > len(claims):
                end_i = len(claims)

            claims_batch = claims[start_i:end_i]
            list_sents_batch = sents[start_i:end_i]

            h_batch, h_lengths_batch = self.padding(claims_batch, self.word_dict, self.h_max_length)
            s_batch, s_lengths_batch = self.padding(list_sents_batch, self.word_dict, self.s_max_length)
            # print(h_batch.shape)

            yield h_batch, h_lengths_batch, s_batch, s_lengths_batch

    def evaluate(self, devs, dev_labels, at=5):

        # c_1_j = 0
        c_2_j = 0

        # count1 = 0
        for i, dev in tqdm(enumerate(devs)):

            labels = dev_labels[i]

            predictions = []
            for h_batch, h_lengths_batch, s_batch, s_lengths_batch in self.generate_dev_batch(dev):
                if len(h_batch) == 0 or len(s_batch) == 0:
                    continue
                feed_dict = {self._X_h: h_batch, self._X_s: s_batch, self._X_h_length: h_lengths_batch,
                             self._X_s_length: s_lengths_batch}
                predicts = self._session.run(self.scores, feed_dict=feed_dict)
                predicts = np.reshape(predicts, newshape=(-1,))
                predicts = predicts.tolist()
                predictions.extend(predicts)
            predictions = np.reshape(np.asarray(predictions), newshape=(-1,))

            rank_index = np.argsort(predictions).tolist()[::-1][:at]

            # score = 0.0
            count = 0.0
            # for i in range(1,len(rank_index)+1):
            #     if labels[rank_index[i-1]] == 1:
            #         count += 1
            #         score += count/i
            for i in range(1, len(rank_index) + 1):
                if labels[rank_index[i - 1]] == 1:
                    count += 1.0
            c_2_j += count / float(sum(labels)) if sum(labels) < 6 else count / float(5)

            # if count == 0.0:
            #     c_1_j = 0.0
            # else:
            #     c_1_j += score/count

        # MAP = c_1_j/float(len(devs))
        overall_accuracy = c_2_j / float(len(devs))

        return overall_accuracy

    def fit(self, X, devs, dev_labels):
        """
        fit the dataset to model for training, if valid is not None, it will report performance of the model in each epoch,
        If the max_check_without_progress is set, the early stopping will be used
        :param X:
        :param y:
        :param valid_X:
        :param y_valid:
        :return:
        """

        self._graph = tf.Graph()
        # random.seed(self.random_state)

        with self._graph.as_default():
            self._construct_graph()

        best_accuaracy = 0
        best_parameters = None
        check_without_progress = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self._session = tf.Session(graph=self._graph, config=config)

        with self._session.as_default() as sess:
            self._init.run()

            for epoch in range(self.num_epoch):
                losses = []

                # random.shuffle(X)
                start_time = time.time()
                for batch_i, (X_h_batch, X_s_batch, X_h_lengths_batch, X_s_lengths_batch) in enumerate(
                        self.generate_batch(X)):
                    # print(batch_i)
                    feed_dict = {self._X_h: X_h_batch, self._X_s: X_s_batch, self._X_h_length: X_h_lengths_batch,
                                 self._X_s_length: X_s_lengths_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True

                    _, loss = sess.run([self._training_ops, self._loss], feed_dict=feed_dict)
                    losses.append(loss)

                current_loss = sum(losses) / len(losses)
                accuracy = self.evaluate(devs, dev_labels)
                end_time = time.time()
                print(
                    "Epoch: {}, Time Costs: {},Current training loss: {:.6f}, Current dev overall accuracy:{:.6f}".format(
                        epoch, (end_time - start_time), current_loss, accuracy))

                if accuracy > best_accuaracy:
                    best_accuaracy = accuracy
                    best_parameters = self._get_model_parameters()
                    check_without_progress = 0
                else:
                    check_without_progress += 1

                if check_without_progress > self.max_checks_without_progress:
                    print(
                        "Stopping Early! Loss has not improved in {} epoches".format(self.max_checks_without_progress))
                    break

            if best_parameters is not None and self.model_store_dir is not None:
                self._restore_model_parameters(best_parameters)
                save_path = os.path.join(self.model_store_dir, "best_model.ckpt")
                self._saver.save(sess, save_path)
                return self

    def restore_model(self, path):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._construct_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = float(os.getenv("TF_GPU_MEMORY_FRACTION","0.33"))

        self._session = tf.Session(config=config, graph=self._graph)

        with self._graph.as_default():
            with tf.variable_scope("embedding_lookup", reuse=True):
                v = tf.get_variable("embedding")
                self._session.run(v.initializer)

        self._saver.restore(self._session, path)
        return self

    def predict(self, X):

        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)

        with self._session.as_default() as sess:
            predicts = []
            claims = [claim for claim, _ in X]
            sents = [sent for _, sent in X]
            for i in range(ceil(len(claims) / self.batch_size)):
                start_i = i * self.batch_size
                end_i = (i + 1) * self.batch_size
                if start_i == len(claims):
                    break
                if end_i > len(claims):
                    end_i = len(claims)
                X_h_batch = claims[start_i:end_i]
                X_s_batch = sents[start_i:end_i]
                X_h_batch, X_h_lengths_batch = self.padding(X_h_batch, self.word_dict, self.h_max_length)
                X_s_batch, X_s_lengths_batch = self.padding(X_s_batch, self.word_dict, self.s_max_length)
                feed_dict = {self._X_h: X_h_batch, self._X_s: X_s_batch,
                             self._X_h_length: X_h_lengths_batch,
                             self._X_s_length: X_s_lengths_batch}

                predict = sess.run(self.scores, feed_dict=feed_dict)
                predicts.extend(predict)
            predicts = np.array(predicts)
        return predicts

    def save(self, path):
        self._saver.save(self._session, path)
