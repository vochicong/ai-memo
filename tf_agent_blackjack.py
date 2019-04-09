# %% [markdown]
# # TF Agents の環境

# %%
from __future__ import absolute_import, division, print_function
# !which python
# !sudo apt install -y cuda-cublas-10-0  cuda-cusolver-10-0 cuda-cudart-10-0 cuda-cusparse-10-0
# !conda install -y -c anaconda cudatoolkit
# !pip install tf-nightly-gpu tf-agents-nightly 'gym==0.10.11'
from tf_agents.environments import utils
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import time_step
from tf_agents.specs import array_spec

tf.compat.v1.enable_v2_behavior()


class BlackJackEnv(py_environment.PyEnvironment):
    ACT_GET_CARD = 0
    ACT_END_GAME = 1
    LIMIT_STATE = 21

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1,
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0,
            name='observation'
        )
        self._state = 0
        self._episode_ended = False
        return

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return time_step.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action == self.ACT_END_GAME:
            self._episode_ended = True
        elif action == self.ACT_GET_CARD:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        #   print("New card: {}, Sum: {}".format(new_card, self._state))
        else:
            raise ValueError("`action` should be {} or {}".format(
                self.ACT_GET_CARD, self.ACT_END_GAME
            ))

        if self._episode_ended or self._state >= self.LIMIT_STATE:
            reward = self._state if self._state <= self.LIMIT_STATE else 0
            # print("End of Game. Rewarded", reward)
            return time_step.termination(
                np.array([self._state], dtype=np.int32), reward)

        return time_step.transition(
            np.array([self._state], dtype=np.int32),
            reward=0.0,
            discount=1.0)


def print_spec(env):
    act_spec, ts_spec = env.action_spec(), env.time_step_spec()
    for x in (act_spec, ts_spec.observation, ts_spec.step_type,
              ts_spec.discount, ts_spec.reward):
        print(x)
    return


def play_blackjack(
    env, n_cards=3,
    act_get_card=lambda: tf.constant([BlackJackEnv.ACT_GET_CARD]),
    act_end_game=lambda: tf.constant([BlackJackEnv.ACT_END_GAME]),
):
    ts = env.reset()
    gain = ts.reward
    cards = []
    for _ in range(n_cards):
        if ts.is_last():
            break
        ts = env.step(act_get_card())
        # print(ts)
        cards += [ts.observation[0][0].numpy()]
        gain += ts.reward

    if not ts.is_last():
        ts = env.step(act_end_game())
        # print(ts)
        gain += ts.reward
    gain = gain.numpy()[0]
    # print("Cards: {}. Reward: {}.".format(cards, gain))
    return cards, gain


env = BlackJackEnv()
# utils.validate_py_environment(env)
# print_spec(env)
# play_blackjack(
#     env,
#     act_get_card=lambda: BlackJackEnv.ACT_GET_CARD,
#     act_end_game=lambda: BlackJackEnv.ACT_END_GAME,
# )

env = tf_py_environment.TFPyEnvironment(env)
gains = []
for _ in range(100):
    _, gain = play_blackjack(env)
    gains += [gain]
np.mean(gains)  # about 12
# %%
help(time_step.TimeStep)
