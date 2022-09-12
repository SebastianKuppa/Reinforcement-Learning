from random import randint, choice
from collections import deque

import pandas as pd
import pygame
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.layers import Input, BatchNormalization, GlobalMaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import time
import random

import models_arch
import constants as const
import game_components.environment

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

MINIBATCH_SIZE = 0
PATH = ""
# create model folder
if not os.path.isdir(f'{PATH}models'):
    os.makedirs(f'{PATH}models')
# create results folder
if not os.path.isdir(f'{PATH}results'):
    os.makedirs(f'{PATH}results')

pygame.init()


class DQNAgent:
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
        self.tensorboard = ModifiedTensorBoard(log_dir="{}logs/{}-{}".format(PATH, name, int(time.time())))

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
        current_qs_list = self.model.predict(current_states.reshape(-1, *self.env.ENVIRONMENT_SHAPE))

        # get future states from batch then query Q vals
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.model.predict(new_current_states.reshape(-1, *self.env.ENVIRONMENT_SHAPE))

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

        self.model.fit(x=np.array(x).reshape(-1, *self.env.ENVIRONMENT_SHAPE),
                       y=np.array(y),
                       batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

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
        return self.model.predict(state.reshape(-1, *self.env.ENVIRONMENT_SHAPE))


# safe model and return model weights
def safe_models_and_weight(agent, model_name, episode, max_reward, average_reward, min_reward):
    checkpoint_name = f"{model_name} | Eps({episode}) | max({max_reward:_>7.2f}) | " \
                      f"avg({average_reward:_>7.2f}) | min({min_reward:_>7.2f}).model"
    agent.model.save(f'{PATH}models/{checkpoint_name}')
    best_weights = agent.model.get_weights()
    return best_weights


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()


def grid_search():
    for i, m in enumerate(models_arch.models_arch):
        start_time = time.time()
        MINIBATCH_SIZE = m["MINIBATCH_SIZE"]

        # exploration settings
        EF_Enabled = m["EF_Settings"]["EF_Enabled"]
        MAX_EPSILON = 1
        MIN_EPSILON = .001
        if EF_Enabled:
            FLUCTUATIONS = m["EF_Settings"]["FLUCTUATIONS"]
            FLUCTUATE_EVERY = int(const.EPISODES/FLUCTUATIONS)
            EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON/FLUCTUATE_EVERY)
            epsilon = 1     # will decay
        else:
            EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON/(.8*const.EPISODES))
            epsilon = 1     # will decay

        # init average and score
        best_average = -100
        best_score = -100

        ECC_Enabled = m["ECC_Settings"]["ECC_Enabled"]
        avg_reward_info = [[1, best_average, epsilon]]
        max_reward_info = [[1, best_score, epsilon]]

        if ECC_Enabled:
            MAX_EPS_NO_INC = m["ECC_Settings"]["MAX_EPS_NO_INC"]

        # episodes without increment reward
        eps_np_inc_counter = 0

        ep_rewards = [best_average]

        # init environment
        env = game_components.environment.Environment()

        env.MOVE_WALL_EVERY = 1

        agent = DQNAgent(name=f"M{i}",
                         env=env,
                         conv_list=m["conv_list"],
                         dense_list=m["dense_list"],
                         util_list=m["util_list"])
        MODEL_NAME = agent.name

        best_weights = [agent.model.get_weights()]

        # Uncomment these two lines if you want to show preview on your screen
        # WINDOW          = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
        # clock           = pygame.time.Clock()

        for episode in tqdm(range(1, const.EPISODES+1), ascii=True, unit="episodes"):
            if m["best_only"]:
                agent.model.set_weights(best_weights[0])

            score_increased = False
            # update tensorboard
            agent.tensorboard.step = episode

            # restart episode values
            episode_reward = 0
            step = 1
            action = 0

            # reset environment and get values
            current_state = env.reset()
            game_over = env.game_over

            while not game_over:
                if np.random.random() > epsilon:
                    # get q action
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    # rnd action
                    action = choice(env.ACTION_SPACE)

                new_state, reward, game_over = env.step(action)

                episode_reward += reward

                # Uncomment the next block if you want to show preview on your screen
                # if SHOW_PREVIEW and not episode % SHOW_EVERY:
                #     clock.tick(27)
                #     env.render(WINDOW)

                agent.update_replay_memory((current_state, action, reward, new_state, game_over))
                agent.train(game_over, step)

                current_state = new_state
                step += 1

            if ECC_Enabled:
                eps_np_inc_counter += 1

            # append episode reward
            ep_rewards.append(episode_reward)

            if not episode % const.AGGREGATE_STATS_EVERY:
                average_reward = \
                    sum(ep_rewards[-const.AGGREGATE_STATS_EVERY:])/len(ep_rewards[const.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-const.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-const.AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward,
                                               reward_min=min_reward,
                                               reward_max=max_reward,
                                               epsilon=epsilon)
                # save models when avg is >= set value
                if not episode % const.SAVE_MODEL_EVERY:
                    _ = safe_models_and_weight(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)
                if average_reward > best_average:
                    best_average = average_reward
                    # update ECC vars
                    avg_reward_info.append([episode, best_average, epsilon])
                    eps_np_inc_counter = 0
                    # save agent
                    best_weights[0] = safe_models_and_weight(agent,
                                                             MODEL_NAME,
                                                             episode,
                                                             max_reward,
                                                             average_reward,
                                                             min_reward)
                if ECC_Enabled and eps_np_inc_counter >= MAX_EPS_NO_INC:
                    epsilon = avg_reward_info[-1][2]
                    eps_np_inc_counter = 0
            if episode_reward > best_score:
                try:
                    best_score = episode_reward
                    max_reward_info.append([episode, best_score, epsilon])
                    best_weights[0] = safe_models_and_weight(agent,
                                                             MODEL_NAME,
                                                             episode,
                                                             max_reward,
                                                             average_reward,
                                                             min_reward)
                except:
                    pass

            # epsilon decay
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            # epsilon fluctuation
            if EF_Enabled:
                if not episode % FLUCTUATE_EVERY:
                    epsilon = MAX_EPSILON

        end_time = time.time()
        total_train_time_sec = round((end_time - start_time))
        total_train_time_min = round((end_time - start_time)/60, 2)
        time_per_episode_sec = round(total_train_time_sec/const.EPISODES, 3)

        # avg reward
        average_reward = round(sum(ep_rewards)/len(ep_rewards), 2)

        # Update Results DataFrames:
        res = res.append(
            {"Model Name": MODEL_NAME, "Convolution Layers": m["conv_list"], "Dense Layers": m["dense_list"],
             "Batch Size": m["MINIBATCH_SIZE"], "ECC": m["ECC_Settings"], "EF": m["EF_Settings"],
             "Best Only": m["best_only"], "Average Reward": average_reward,
             "Best Average": avg_reward_info[-1][1], "Epsilon 4 Best Average": avg_reward_info[-1][2],
             "Best Average On": avg_reward_info[-1][0], "Max Reward": max_reward_info[-1][1],
             "Epsilon 4 Max Reward": max_reward_info[-1][2], "Max Reward On": max_reward_info[-1][0],
             "Total Training Time (min)": total_train_time_min, "Time Per Episode (sec)": time_per_episode_sec}
            , ignore_index=True)
        res = res.sort_values(by="Best Average")

        avg_df = pd.DataFrame(data=avg_reward_info, columns=["Episode", "Average Reward", "Epsilon"])
        max_df = pd.DataFrame(data=max_reward_info, columns=["Episode", "Max Reward", "Epsilon"])

        # save dataframes
        res.to_csv(f'{PATH}results/Results.csv')
        avg_df.to_csv(f'{PATH}results/{MODEL_NAME}-Results-Avg.csv')
        max_df.to_csv(f'{PATH}results/{MODEL_NAME}-Results-Max.csv')

