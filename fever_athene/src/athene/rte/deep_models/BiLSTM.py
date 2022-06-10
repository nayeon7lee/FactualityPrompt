import tensorflow as tf

from athene.rte.deep_models.LSTM import LSTM

he_init = tf.contrib.layers.variance_scaling_initializer()


class BiLSTM(LSTM):

    def __init__(self, h_max_length=20, b_max_length=200, trainable=False, lstm_layers=2, mlp_layers=1,
                 num_neurons=[128, 128, 32], share_parameters=True, average_pooling=False,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.001, batch_size=128, activation=tf.nn.relu, initializer=he_init, num_epoch=20,
                 batch_norm_momentum=None, dropout_rate=None,
                 max_check_without_progress=20, show_progress=10, tensorboard_logdir=None, random_state=None,
                 embedding=None, l2_lambda=0.01):
        LSTM.__init__(self, h_max_length, b_max_length, trainable, lstm_layers, mlp_layers, num_neurons,
                      share_parameters, average_pooling, optimizer, learning_rate, batch_size, activation, initializer,
                      num_epoch, batch_norm_momentum, dropout_rate,
                      max_check_without_progress, show_progress, tensorboard_logdir, random_state, embedding, l2_lambda)

    def _share_lstm(self, inputs, sequence_length):

        if self.lstm_layers == 1:
            rnn_cells_fw = self.lstm_cell(self.num_neurons[0])
            rnn_cells_bw = self.lstm_cell(self.num_neurons[0])
        elif self.lstm_layers > 1:
            rnn_cells_fw = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])
            rnn_cells_bw = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])

        inputs = tf.cast(inputs, tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cells_fw, rnn_cells_bw, inputs=inputs, dtype=tf.float32,
                                                          sequence_length=sequence_length)

        if self.average_pooling:
            output_fw = tf.reduce_mean(outputs[0], axis=1, name="average_fw")
            output_bw = tf.reduce_mean(outputs[1], axis=1, name="average_bw")
            output = tf.concat([output_fw, output_bw], axis=1)
        if self.lstm_layers == 1:
            output_fw = states[0][1]
            output_bw = states[1][1]
            output = tf.concat([output_fw, output_bw], axis=1)
        elif self.lstm_layers > 1:
            output_fw = states[self.lstm_layers - 1][0][1]
            output_bw = states[self.lstm_layers - 1][1][1]
            output = tf.concat([output_fw, output_bw], axis=1)

        return output

    def _separate_lstm(self, heads_inputs, body_inputs, heads_length, bodies_length):

        if self.lstm_layers == 1:
            heads_rnn_cells_fw = self.lstm_cell(self.num_neurons[0])
            heads_rnn_cells_bw = self.lstm_cell(self.num_neurons[0])
            bodies_rnn_cells_fw = self.lstm_cell(self.num_neurons[0])
            bodies_rnn_cells_bw = self.lstm_cell(self.num_neurons[0])
        elif self.lstm_layers > 1:
            heads_rnn_cells_fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])
            heads_rnn_cells_bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])
            bodies_rnn_cells_fw = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])
            bodies_rnn_cells_bw = tf.nn.rnn_cell.MultiRNNCell(
                [self.lstm_cell(i) for i in self.num_neurons[:self.lstm_layers]])

        heads_inputs = tf.cast(heads_inputs, tf.float32)
        body_inputs = tf.cast(body_inputs, tf.float32)
        with tf.variable_scope("rnn_heads"):
            heads_outputs, heads_states = tf.nn.bidirectional_dynamic_rnn(heads_rnn_cells_fw, heads_rnn_cells_bw,
                                                                          inputs=heads_inputs, dtype=tf.float32,
                                                                          sequence_length=heads_length)
        with tf.variable_scope("rnn_bodies"):
            bodies_outputs, bodies_states = tf.nn.bidirectional_dynamic_rnn(bodies_rnn_cells_fw, bodies_rnn_cells_bw,
                                                                            inputs=body_inputs, dtype=tf.float32,
                                                                            sequence_length=bodies_length)

        heads_outputs = tf.concat(heads_states, axis=2)
        bodies_outputs = tf.concat(bodies_states, axis=2)

        if self.average_pooling:
            head_output_fw = tf.reduce_mean(heads_outputs[0], axis=1, name="avearge_pooling_heads_fw")
            head_output_bw = tf.reduce_mean(heads_outputs[1], axis=1, name="avearge_pooling_heads_bw")
            body_output_fw = tf.reduce_mean(bodies_outputs[0], axis=1, name="average_pooling_bodies_fw")
            body_output_bw = tf.reduce_mean(bodies_outputs[1], axis=1, name="average_pooling_bodies_bw")
            head_output = tf.concat([head_output_fw, head_output_bw], axis=1)
            body_output = tf.concat([body_output_fw, body_output_bw], axis=1)
        else:
            if self.lstm_layers == 1:
                head_output_fw = heads_states[0][1]
                head_output_bw = heads_states[1][1]
                head_output = tf.concat([head_output_fw, head_output_bw], axis=1)
                body_output_fw = bodies_states[0][1]
                body_output_bw = bodies_states[1][1]
                body_output = tf.concat([body_output_fw, body_output_bw], axis=1)
            elif self.lstm_layers > 1:
                head_output_fw = heads_states[self.lstm_layers - 1][0][1]
                head_output_bw = heads_states[self.lstm_layers - 1][1][1]
                head_output = tf.concat([head_output_fw, head_output_bw], axis=1)
                body_output_fw = bodies_states[self.lstm_layers - 1][0][1]
                body_output_bw = bodies_states[self.lstm_layers - 1][1][1]
                body_output = tf.concat([body_output_fw, body_output_bw], axis=1)

        return head_output, body_output
