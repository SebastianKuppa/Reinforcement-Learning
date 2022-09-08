from collections import deque
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, BatchNormalization, GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
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