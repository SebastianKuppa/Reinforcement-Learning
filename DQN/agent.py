from collections import deque
from keras.callbacks import TensorBoard


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
        self.writer =