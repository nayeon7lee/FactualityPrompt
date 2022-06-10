from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from common.util.log_helper import LogHelper

he_init = tf.contrib.layers.variance_scaling_initializer()


class LSTM(BaseEstimator, ClassifierMixin):

    def __init__(self, h_max_length=20, b_max_length=200, trainable=False, lstm_layers=1, mlp_layers=0,
                 num_neurons=[128, 128, 32], share_parameters=True, average_pooling=False,
                 optimizer=tf.train.AdamOptimizer, learning_rate=0.001,
                 batch_size=128, activation=tf.nn.relu, initializer=he_init, num_epoch=10, batch_norm_momentum=None,
                 dropout_rate=None,
                 max_check_without_progress=20, show_progress=10, tensorboard_logdir=None, random_state=None,
                 embedding=None, l2_lambda=0.001, vocab_size=None):
        self.logger = LogHelper.get_logger(self.__class__.__name__)
        self.h_max_length = h_max_length
        self.b_max_length = b_max_length
        self.trainable = trainable
        self.lstm_layers = lstm_layers
        self.mlp_layers = mlp_layers
        self.num_neurons = num_neurons
        self.share_parameters = share_parameters
        self.average_pooling = average_pooling
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.num_epoch = num_epoch
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.max_checks_without_progress = max_check_without_progress
        self.show_progress = show_progress
        self.randome_state = random_state
        self.tensorboard_logdir = tensorboard_logdir
        self.embedding = embedding
        self.embed_size = len(embedding[0]) if embedding is not None else 0
        self.l2_lambda = l2_lambda
        self.logger.debug(vocab_size)
        self.vocab_size = vocab_size
        # self.share_embeddings = share_embeddings

        # assert self.lstm_layers + self.mlp_layers == len(self.num_neurons)

        # if self.embedding is None and self.vocab_size is None:
        #     raise Exception("Either embedding or vocab_size must be setted!")

        self._session = None

    def lstm_cell(self, hidden_size):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        if self.dropout_rate:
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        return lstm

    def add_embedding(self, heads_inputs, bodies_inputs, trainable=False):
        """Adds an embedding layer that maps from input tokens (integers) to vectors for both the headings and bodies:

        Returns:
            embeddings_headings: tf.Tensor of shape (None, h_max_len, embed_size)
            embeddings_bodies: tf.Tensor of shape (None, b_max_len, embed_size)
        """

        if not trainable:
            embeddings_headings = tf.nn.embedding_lookup(params=tf.constant(self.embedding), ids=heads_inputs)
            embeddings_bodies = tf.nn.embedding_lookup(params=tf.constant(self.embedding), ids=bodies_inputs)
        else:
            embeddings_headings = tf.nn.embedding_lookup(params=tf.Variable(self.embedding), ids=heads_inputs)
            embeddings_bodies = tf.nn.embedding_lookup(params=tf.Variable(self.embedding), ids=bodies_inputs)

        embeddings_headings = tf.reshape(embeddings_headings, shape=(-1, self.h_max_length, self.embed_size))
        embeddings_bodies = tf.reshape(embeddings_bodies, shape=(-1, self.b_max_length, self.embed_size))

        return embeddings_headings, embeddings_bodies

    def add_embedding_share(self, inputs, max_length, trainable=False):

        embeddings = tf.nn.embedding_lookup(params=tf.Variable(self.embedding, trainable=trainable), ids=inputs)
        embeddings = tf.reshape(embeddings, shape=(-1, max_length, self.embed_size))

        return embeddings

    # def extract_axis_1(data, ind):
    #     """
    #     Get specified elements along the first axis of tensor.
    #     :param data: Tensorflow tensor that will be subsetted.
    #     :param ind: Indices to take (one for each element along axis 0 of data).
    #     :return: Subsetted tensor.
    #     """
    #
    #     batch_range = tf.range(tf.shape(data)[0])
    #     indices = tf.stack([batch_range, ind], axis=1)
    #     res = tf.gather_nd(data, indices)
    #
    #     return res

    def _share_lstm(self, inputs, sequence_length):

        inputs = tf.cast(inputs, tf.float32)
        if self.lstm_layers < 1:
            raise NotImplementedError("No implementation of lstm cell is not allowed in lstm model.")

        if self.lstm_layers == 1:
            rnn_cells = self.lstm_cell(self.num_neurons[0])
        elif self.lstm_layers >= 1:
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])

        outputs, state = tf.nn.dynamic_rnn(rnn_cells, inputs=inputs, dtype=tf.float32, sequence_length=sequence_length)
        if self.average_pooling:
            output = tf.reduce_mean(outputs, axis=1, name="average_pooling")
            # self.logger.debug(output.get_shape())
        else:
            if self.lstm_layers == 1:
                output = state[1]
            elif self.lstm_layers > 1:
                output = state[self.lstm_layers - 1][1]

        return output

    def _separate_lstm(self, heads_inputs, body_inputs, heads_length, bodies_length):

        heads_inputs = tf.cast(heads_inputs, tf.float32)
        body_inputs = tf.cast(body_inputs, tf.float32)
        if self.lstm_layers < 1:
            raise NotImplementedError("No implementation of lstm cell is not allowed in lstm model.")
        if self.lstm_layers == 1:
            heads_rnn_cells = self.lstm_cell(self.num_neurons[0])
            bodies_rnn_cells = self.lstm_cell(self.num_neurons[0])
        elif self.lstm_layers > 1:
            heads_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])
            bodies_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])

        with tf.variable_scope("rnn_heads"):
            heads_outputs, heads_states = tf.nn.dynamic_rnn(heads_rnn_cells, inputs=heads_inputs, dtype=tf.float32,
                                                            sequence_length=heads_length)
        with tf.variable_scope("rnn_bodies"):
            bodies_outputs, bodies_states = tf.nn.dynamic_rnn(bodies_rnn_cells, inputs=body_inputs, dtype=tf.float32,
                                                              sequence_length=bodies_length)

        if self.average_pooling:
            head_output = tf.reduce_mean(heads_outputs, axis=1, name="avearge_pooling_heads")
            body_output = tf.reduce_mean(bodies_outputs, axis=1, name="average_pooling_bodies")
        else:
            if self.lstm_layers == 1:
                head_output = heads_states[1]
                body_output = bodies_states[1]
            elif self.lstm_layers > 1:
                head_output = heads_states[self.lstm_layers - 1][1]
                body_output = bodies_states[self.lstm_layers - 1][1]
        return head_output, body_output

    def _dnn(self, head_inputs, body_inputs, heads_length, bodies_length):

        head_inputs = self.add_embedding_share(head_inputs, self.h_max_length, trainable=self.trainable)
        body_inputs = self.add_embedding_share(body_inputs, self.b_max_length, trainable=self.trainable)

        if self.share_parameters:
            with tf.variable_scope("lstm_lower") as scope:
                output_head = self._share_lstm(head_inputs, heads_length)
                scope.reuse_variables()
                output_body = self._share_lstm(body_inputs, bodies_length)
        else:
            output_head, output_body = self._separate_lstm(head_inputs, body_inputs, heads_length, bodies_length)

        inputs = tf.concat([output_head, output_body], axis=1)

        for layer in range(self.mlp_layers):

            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.num_neurons[self.lstm_layers + layer], activation=self.activation,
                                     kernel_initializer=self.initializer,
                                     name="hidden{}".format(layer + 1))

            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)

            inputs = self.activation(inputs, name="hidden{}_out".format(layer + 1))

        return inputs

    def _construct_graph(self, n_outputs, weights):

        if self.randome_state:
            tf.set_random_seed(self.randome_state)
            np.random.seed(self.randome_state)

        X_heads = tf.placeholder(tf.int32, shape=[None, self.h_max_length], name="X_heads")
        X_bodies = tf.placeholder(tf.int32, shape=[None, self.b_max_length], name="X_bodies")
        X_head_length = tf.placeholder(tf.int32, shape=[None], name="head_length")
        X_body_length = tf.placeholder(tf.int32, shape=[None], name="body_length")
        y_ = tf.placeholder(tf.int32, shape=[None], name="y")
        weights = tf.constant(weights, tf.float32)

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=[], name="training")
            self.keep_prob = tf.cond(self._training, lambda: tf.constant(1 - self.dropout_rate),
                                     lambda: tf.constant(1.0))
        else:
            self._training = None

        pre_output = self._dnn(X_heads, X_bodies, X_head_length, X_body_length)
        logits = tf.layers.dense(pre_output, n_outputs, kernel_initializer=he_init, name="logits")
        # weighted_logits = tf.multiply(logits,weights)
        probabilities = tf.nn.softmax(logits, name="probabilities")
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        variables = tf.trainable_variables()
        for v in variables:
            self.logger.debug(v.name)
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in variables if 'bias' not in v.name and 'Variable' not in v.name]) * self.l2_lambda
        loss += l2_loss

        optimizer = self.optimizer(learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y_, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        if self.tensorboard_logdir:
            now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            tb_logdir = self.tensorboard_logdir + "/run{}".format(now)
            cost_summary = tf.summary.scalar("validation_loss", loss)
            acc_summary = tf.summary.scalar("validation_accuracy", accuracy)
            merged_summary = tf.summary.merge_all()
            file_writer = tf.summary.FileWriter(tb_logdir, tf.get_default_graph())

            self._merged_summary = merged_summary
            self._file_writer = file_writer

        self._X_head, self._X_body, self._X_head_length, self._X_body_length, self.y = X_heads, X_bodies, X_head_length, X_body_length, y_
        self._logits = logits
        self._probabilites = probabilities
        self._loss = loss
        self._training_op = training_op
        self._accuracy = accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_parameters(self):

        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_parameters(self, model_parameters):

        gvar_names = list(model_parameters.keys())

        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}

        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_parameters[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def get_truncted_data(self, h_np, b_np, h_lenseq, b_lenseq):

        if self.h_max_length:
            if h_np.shape[1] > self.h_max_length:
                h_np = h_np[:, :self.h_max_length]

            h_lenseq = np.minimum(h_lenseq, self.h_max_length)

        if self.b_max_length:
            if b_np.shape[1] > self.b_max_length:
                b_np = b_np[:, :self.b_max_length]
            b_lenseq = np.minimum(b_lenseq, self.b_max_length)
        return h_np, b_np, h_lenseq, b_lenseq

    def get_batch(self, h_np, b_np, h_lenseq, b_lenseq, y):

        num_batches = h_np.shape[0] // self.batch_size
        batches = []

        for i in range(num_batches):
            if (i + 1) * self.batch_size > h_np.shape[0]:
                batch_h_np = h_np[i * self.batch_size:h_np.shape[0] - 1, :]
                batch_b_np = b_np[i * self.batch_size:h_np.shape[0] - 1, :]
                batch_h_lenseq = h_lenseq[i * self.batch_size:h_np.shape[0] - 1]
                batch_b_lenseq = b_lenseq[i * self.batch_size:h_np.shape[0] - 1]
                batch_y = y[i * self.batch_size:h_np.shape[0] - 1]
            else:
                batch_h_np = h_np[i * self.batch_size:(i + 1) * self.batch_size, :]
                batch_b_np = b_np[i * self.batch_size:(i + 1) * self.batch_size, :]
                batch_h_lenseq = h_lenseq[i * self.batch_size:(i + 1) * self.batch_size]
                batch_b_lenseq = b_lenseq[i * self.batch_size:(i + 1) * self.batch_size]
                batch_y = y[i * self.batch_size:(i + 1) * self.batch_size]
            batches.append((batch_h_np, batch_b_np, batch_h_lenseq, batch_b_lenseq, batch_y))

        return batches

    def pad_data(self, h_np, b_np):

        if h_np.shape[1] < self.h_max_length:
            new_h_np = np.zeros(h_np.shape[0], self.h_max_length)
            for idx, vec in enumerate(h_np):
                new_h_np[idx, :h_np.shape[1]] = vec
            h_np = new_h_np

        if b_np.shape[1] < self.b_max_length:
            new_b_np = np.zeros(b_np.shape[0], self.b_max_length)
            for dix, vec in enumerate(b_np):
                new_b_np[idx, :b_np.shape[0]] = vec
            b_np = new_b_np

        return h_np, b_np

    def fit(self, X, y, valid_X=None, y_valid=None):

        self.close_session()
        y = np.array(y)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self._classes, counts = np.unique(y, return_counts=True)
        weights = [count / sum(counts) for count in counts]
        weights = np.asarray(weights, np.float32)
        n_outputs = len(self._classes)

        self._graph = tf.Graph()
        h_np, b_np, h_lenseq, b_lenseq = X['h_np'], X['b_np'], X['h_seqlen'], X['b_seqlen']

        with self._graph.as_default():
            self._construct_graph(n_outputs, weights)

        checks_without_progress = 0
        best_acc = np.inf
        best_parameters = None
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self._session = tf.Session(graph=self._graph, config=tf.ConfigProto(gpu_options=gpu_options))

        if h_np.shape[1] < self.h_max_length or b_np.shape[1] < self.b_max_length:
            raise ValueError("Selected claim or evidence length if less that its largest length!")

        h_np, b_np, h_lenseq, b_lenseq = self.get_truncted_data(h_np, b_np, h_lenseq, b_lenseq)

        with self._session.as_default() as sess:
            self._init.run()
            num_instances = h_np.shape[0]
            for epoch in range(self.num_epoch):
                losses = []
                accs = []
                rnd_idx = np.random.permutation(num_instances)
                for rnd_indices in np.array_split(rnd_idx, num_instances // self.batch_size):

                    X_head_batch, X_body_batch, X_head_len, X_body_len, y_batch = h_np[rnd_indices], b_np[rnd_indices], \
                                                                                  h_lenseq[rnd_indices], b_lenseq[
                                                                                      rnd_indices], y[rnd_indices]

                    y_batch = np.asarray(y_batch)
                    feed_dict = {self._X_head: X_head_batch, self._X_body: X_body_batch,
                                 self._X_head_length: X_head_len, self._X_body_length: X_body_len, self.y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True

                    train_acc, _, loss = sess.run([self._accuracy, self._training_op, self._loss], feed_dict=feed_dict)
                    losses.append(loss)
                    accs.append(train_acc)
                average_loss = sum(losses) / len(losses)
                average_acc = sum(accs) / len(accs)

                if valid_X is not None and y_valid is not None:

                    self.logger.info("validation phase")
                    valid_h_np, valid_b_np, valid_h_lenseq, valid_b_lenseq = valid_X['h_np'], valid_X['b_np'], valid_X[
                        'h_seqlen'], valid_X['b_seqlen']

                    valid_h_np, valid_b_np, valid_h_lenseq, valid_b_lenseq = self.get_truncted_data(valid_h_np,
                                                                                                    valid_b_np,
                                                                                                    valid_h_lenseq,
                                                                                                    valid_b_lenseq)
                    if valid_h_np.shape[1] < self.h_max_length or valid_b_np.shape[1] < self.b_max_length:
                        valid_h_np, valid_b_np = self.pad_data(valid_h_np, valid_b_np)
                    batches = self.get_batch(valid_h_np, valid_b_np, valid_h_lenseq, valid_b_lenseq, y_valid)

                    batch_losses = []
                    batch_accuracies = []
                    for (valid_h_batch, valid_b_batch, valid_h_len_batch, valid_b_len_batch, valid_y_batch) in batches:

                        feed_dict_valid = {self._X_head: valid_h_batch, self._X_body: valid_b_batch,
                                           self._X_head_length: valid_h_len_batch,
                                           self._X_body_length: valid_b_len_batch, self.y: valid_y_batch}
                        if self.tensorboard_logdir:
                            val_acc_batch, val_loss_batch, summary = sess.run(
                                [self._accuracy, self._loss, self._merged_summary], feed_dict=feed_dict_valid)
                            self._file_writer.add_summary(summary, epoch)
                        else:
                            val_acc_batch, val_loss_batch = sess.run([self._accuracy, self._loss],
                                                                     feed_dict=feed_dict_valid)

                        batch_losses.append(val_loss_batch)
                        batch_accuracies.append(val_acc_batch)

                    val_loss = sum(batch_losses) / len(batch_losses)
                    val_acc = sum(batch_accuracies) / len(batch_accuracies)
                    if self.show_progress:
                        if epoch % self.show_progress == 0:
                            self.logger.info(
                                "Epoch: {} Current training accuracy: {:.4f} ,Current training loss: {:.6f} Validation Accuracy: {:.4f} Validation Loss{:.6f}".format(
                                    epoch + 1, average_acc, average_loss, val_acc, val_loss))

                    if val_acc > best_acc:
                        best_acc = val_acc
                        checks_without_progress = 0
                        self.logger.info("accuracy has been improved!")
                        best_parameters = self._get_model_parameters()
                    else:
                        checks_without_progress += 1

                    if checks_without_progress > self.max_checks_without_progress:
                        self.logger.info("Stopping Early! Loss has not improved in {} epoches".format(
                            self.max_checks_without_progress))
                        break
                else:
                    if self.show_progress:
                        if epoch % self.show_progress == 0:
                            self.logger.info("Epoch: {} Current training accuracy: {:.4f}".format(epoch + 1, train_acc))

            if best_parameters:
                self._restore_model_parameters(best_parameters)
                return self

    def predict_probabilites(self, X):

        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)

        h_np, b_np, h_lenseq, b_lenseq = X['h_np'], X['b_np'], X['h_seqlen'], X['b_seqlen']
        h_np, b_np, h_lenseq, b_lenseq = self.get_truncted_data(h_np, b_np, h_lenseq, b_lenseq)
        if h_np.shape[1] < self.h_max_length or b_np.shape[1] < self.b_max_length:
            h_np, b_np = self.pad_data(h_np, b_np)

        with self._session.as_default() as sess:
            return self._probabilites.eval(
                feed_dict={self._X_head: h_np, self._X_body: b_np, self._X_head_length: h_lenseq,
                           self._X_body_length: b_lenseq})

    def predict(self, X):

        predictions = np.argmax(self.predict_probabilites(X), axis=1)
        return np.reshape(predictions, (-1,))

    def save(self, path):
        self._saver.save(self._session, path)
