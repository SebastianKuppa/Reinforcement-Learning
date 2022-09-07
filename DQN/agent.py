from collections import deque
from keras.callbacks import TensorBoard
import tensorflow as tf
import os


class Agent:
    def __init__(self, name, env, conv_list, dense_list, util_list):
        self.env = env
        self.conv_list = conv_list
        self.dense_list = dense_list
        self.name = [str(name) + " | " + "".join(str(c) + "C | " for c in conv_list)
                                       + "".join(str(d) + "D | " for d in dense_list)
                                       + "".join(u + " | " for u in util_list)][0]
        self.model = self.create_model(self.conv_list, self.dense_list)
        self.target_model = self.create_model(self.conv_list, self.dense_list)

        # collection of the last states and actions
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # init tensorboard for analysis
        self.tensorboard = ModifiedTensorBoard


    # model init
    def create_model(self, conv_list, dense_list):
        return True


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