from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

he_init = tf.contrib.layers.variance_scaling_initializer()


class BaseDeepModel(BaseEstimator, ClassifierMixin):

    def __init__(self, optimizer=tf.train.AdamOptimizer, learning_rate=0.01, initializer=he_init, batch_size=64,
                 activation=tf.nn.relu, num_epoch=100, batch_norm_momentum=None,
                 dropout_rate=None, max_check_without_progress=20, show_progress=10, tensorboard_logdir=None,
                 random_state=None):

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

        self._session = None

    def _dnn(self, head_input, body_inputs):
        raise NotImplementedError('subclass must override this method')

    def _generate_batch_features(self, heads, bodies):
        raise NotImplementedError('subclass must override this method')

    def _construct_graph(self, vector_length, n_outputs):

        if self.randome_state:
            tf.set_random_seed(self.randome_state)
            np.random.seed(self.randome_state)

        X_heads = tf.placeholder(tf.float32, shape=[30, None, vector_length], name="X_heads")
        X_bodies = tf.placeholder(tf.float32, shape=[200, None, vector_length])
        y_ = tf.placeholder(tf.int32, shape=[None], name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=[], name="training")
        else:
            self._training = None

        pre_output = self._dnn(X_heads, X_bodies)
        logits = tf.layers.dense(pre_output, n_outputs, kernel_initializer=he_init, name="logits")
        probabilities = tf.nn.softmax(logits, name="probabilities")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

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

        self._X_head, self._X_body, self.y = X_heads, X_bodies, y_
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

    def fit(self, X, y, valid_X=None, y_valid=None):

        self.close_session()
        n_inputs = self.vector_length
        y = np.array(y)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self._classes = np.unique(y)
        n_outputs = len(self._classes)

        # self.class_to_index_ = {label: index for index,label in enumerate(self._classes)}
        # labels = [self.class_to_index_[label] for label in y]

        self._graph = tf.Graph()

        with self._graph.as_default():
            self._construct_graph(n_inputs, n_outputs)

        checks_without_progress = 0
        best_loss = np.float("inf")
        best_parameters = None
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self._session = tf.Session(graph=self._graph, config=tf.ConfigProto(gpu_options=gpu_options))

        heads, bodies = [], []
        for head, body in X:
            heads.append(head)
            bodies.append(body)
        with self._session.as_default() as sess:
            self._init.run()
            num_instances = len(heads)
            for epoch in range(self.num_epoch):
                total_loss = 0.0
                rnd_idx = np.random.permutation(num_instances)
                for rnd_indices in np.array_split(rnd_idx, num_instances // self.batch_size):

                    X_head_batch, X_body_batch, y_batch = [heads[idx] for idx in rnd_indices], [bodies[idx] for idx in
                                                                                                rnd_indices], [y[idx]
                                                                                                               for idx
                                                                                                               in
                                                                                                               rnd_indices]
                    X_head, X_body = self._generate_batch_features(X_head_batch, X_body_batch)
                    y_batch = np.asarray(y_batch)
                    feed_dict = {self._X_head: X_head, self._X_body: X_body, self.y: y_batch}
                    if self._training is not None:
                        feed_dict = {self._X_head: X_head, self._X_body: X_body, self.y: y_batch, self._training: True}

                    train_acc, _, loss = sess.run([self._accuracy, self._training_op, self._loss], feed_dict=feed_dict)
                    total_loss += loss
                average_loss = total_loss / (num_instances // self.batch_size)

                if valid_X is not None and y_valid is not None:
                    valid_heads, valid_bodies = zip(*valid_X)
                    # batch_losses = []
                    # batch_accuracies = []
                    # for i in range(len(valid_bodies)//self.batch_size):
                    #     head_batch,body_batch,y_batch = valid_heads[i*self.batch_size:(i+1)*self.batch_size],valid_bodies[i*self.batch_size:(i+1)*self.batch_size],y_valid[i*self.batch_size:(i+1)*self.batch_size]
                    #     X_valid_head,X_valid_body = generating_word2vec_batch(self.embedding,head_batch,body_batch,self.vector_length)
                    #     feed_dict_valid = {self._X_head:X_valid_head,self._X_body:X_valid_body,self.y:y_batch}
                    #
                    #     if self.tensorboard_logdir:
                    #         val_acc_batch,val_loss_batch,summary = sess.run([self._accuracy,self._loss,self._merged_summary],feed_dict=feed_dict_valid)
                    #         self._file_writer.add_summary(summary,epoch)
                    #     else:
                    #         val_acc_batch,val_loss_batch = sess.run([self._accuracy,self._loss],feed_dict = feed_dict_valid)
                    #
                    #     batch_losses.append(val_loss_batch)
                    #     batch_accuracies.append(val_acc_batch)
                    #
                    # val_loss = sum(batch_losses)/len(batch_losses)
                    # val_acc = sum(batch_accuracies)/len(batch_accuracies)
                    X_valid_head, X_valid_body = self._generate_batch_features(self.embedding, valid_heads,
                                                                               valid_bodies, vector_length=300)
                    feed_dict_valid = {self._X_head: X_valid_head, self._X_body: X_valid_body, self.y: y_valid}

                    if self.tensorboard_logdir:
                        val_acc, val_loss, summary = sess.run([self._accuracy, self._loss, self._merged_summary],
                                                              feed_dict=feed_dict_valid)
                        self._file_writer.add_summary(summary, epoch)
                    else:
                        val_acc, val_loss = sess.run([self._accuracy, self._loss], feed_dict=feed_dict_valid)

                    if self.show_progress:
                        if epoch % self.show_progress == 0:
                            print(
                                "Epoch: {} Current training accuracy: {:.4f} ,Current training loss: {:.6f} Validation Accuracy: {:.4f} Validation Loss{:.6f}".format(
                                    epoch + 1, train_acc, average_loss, val_acc, val_loss))

                    if val_loss < best_loss:
                        best_loss = val_loss
                        checks_without_progress = 0
                        best_parameters = self._get_model_parameters()
                    else:
                        checks_without_progress += 1

                    if checks_without_progress > self.max_checks_without_progress:
                        print("Stopping Early! Loss has not improved in {} epoches".format(
                            self.max_checks_without_progress))
                        break
                else:
                    if self.show_progress:
                        if epoch % self.show_progress == 0:
                            print("Epoch: {} Current training accuracy: {:.4f}".format(epoch + 1, train_acc))

            if best_parameters:
                self._restore_model_parameters(best_parameters)
                return self

    def predict_probabilites(self, X):

        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)

        heads, bodies = [], []
        for head, body in X:
            heads.append(head)
            bodies.append(body)
        X_heads, X_bodies = self._generate_batch_features(heads, bodies)
        with self._session.as_default() as sess:
            return self._probabilites.eval(feed_dict={self._X_head: X_heads, self._X_body: X_bodies})

    def predict(self, X):

        predictions = np.argmax(self.predict_probabilites(X), axis=1)
        return np.reshape(predictions, (-1,))

    def save(self, path):
        self._saver.save(self._session, path)
