import random
from collections import deque
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, BatchNormalization, GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
import time

import models_arch
import constants as const

PATH = ""


class Agent:
    def __init__(self, name, env, conv_list, dense_list, util_list):
        self.env = env
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.name = [str(name) + " | " + "".join(str(c) + "C | " for c in conv_list)
                                       + "".join(str(d) + "D | " for d in dense_list)
                                       + "".join(u + " | " for u in util_list)][0]
        # init the 2 DQNs for training
        self.model = self.create_model(self.conv_list, self.dense_list)
        self.target_model = self.create_model(self.conv_list, self.dense_list)

        # collection of the last states and actions
        self.replay_memory = deque(maxlen=const.REPLAY_MEMORY_SIZE)

        # init tensorboard for analysis
        self.tensorboard = ModifiedTensorBoard(name, log_dir="{}logs/{}-{}".format(PATH, name, int(time.time())))

        # counter for target network update cycle
        self.target_update_counter = 0

    # init convolutional layer with filters batch_norm, pooling layer and dropout
    def conv_block(self, inp, filters=64, bn=True, pool=True, dropout=.2):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPooling2D(pool_size=(2, 2))(_)
        if dropout > 0:
            _ = Dropout(dropout)(_)

        return _

    # model init
    def create_model(self, conv_list, dense_list):
        # init input layer with array of game environment
        input_layer = Input(shape=self.env.ENVIRONMENT_SHAPE)
        # init 1. conv block
        _ = self.conv_block(input_layer, conv_list[0], bn=False, pool=False)
        # if there are more conv block in the arch, create them in for loop
        if len(conv_list) > 1:
            for c in conv_list[0]:
                _ = self.conv_block(inp=_, filters=c)
        # flatten last layer
        _ = Flatten()(_)

        # dense layers
        for d in dense_list:
            _ = Dense(units=d, activation='relu')(_)
        # 5 layer outputs for the length of the possible action space
        output = Dense(units=self.env.ACTION_SPACE_SIZE, activation='linear', name='output')(_)

        model = Model(inputs=input_layer, outputs=[output])
        model.compile(optimizer=Adam(lr=.001),
                      loss={'output': 'mse'},
                      metrics={'output': 'accuracy'})
        return model

    # update replay memory with new step
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # train the network
    def train(self, terminal_state, step):
        # wait for sufficient amount of replay memory
        if len(self.replay_memory) < const.MIN_REPLAY_MEMORY_SIZE:
            return

        # get mini replay batch
        mini_batch = random.sample(self.replay_memory, models_arch.models_arch[0]["MINIBATCH_SIZE"])

        # get current states from minibatch
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states.reshape(-1, self.env.ENVIRONMENT_SHAPE))

        # get future states from batch then query Q vals
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.model.predict(new_current_states.reshape(-1, self.env.ENVIRONMENT_SHAPE))

        # init training data
        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            # if not terminal state, get new Q from future state, otherwise 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + const.DISCOUNT + max_future_q
            else:
                new_q = reward

            # update q value
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # append to training data
            x.append(current_state)
            y.append(current_qs)

        self.model.fit(x=np.array(x).reshape(-1, self.env.ENVIRONMENT_SHAPE),
                       y=np.array(y),
                       batch_size=models_arch.models_arch[0]["MINIBATCH_SIZE"],
                       shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # update counter
        if terminal_state:
            self.target_update_counter += 1

        # update target model if counter reaches threshold
        if self.target_update_counter > const.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            # reset target counter
            self.target_update_counter = 0

    # get qs from current observation space (meaning environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, self.env.ENVIRONMENT_SHAPE))


def safe_models_and_weight(agent, model_name, episode, max_reward, average_reward, min_reward):
    checkpoint_name = f"{model_name} | Eps({episode}) | max({max_reward:_>7.2f}) | " \
                      f"avg({average_reward:_>7.2f}) | min({min_reward:_>7.2f}).model"
    agent.model.save(f'{PATH}models/{checkpoint_name}')
    best_weights = agent.model.get_weights()
    return best_weights


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.logdir)
        self._log_writer_dir = os.path.join(self.log_dir, name)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, batch, logs=None):
        pass

    def update_stats(self, **stats):
        self.write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()