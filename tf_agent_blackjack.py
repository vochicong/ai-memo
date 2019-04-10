# %% [markdown]
# # TF Agent で Blackjack 遊ぶ
#
# ブラックジャックを参考に、次のような
# ルールとする。
#
# - カードの値は、 1〜11の間にランダムに決まる（エース考慮などはしない）
# - 最初にカードをプレイヤーに2枚、ディーラーに1枚
# - プレイヤーが何枚でもカードを引ける(hit)が、合計が21超えたら即負け。ゲーム終了
# - プレイヤーがカードを引くのを止めたら(stick)、ディーラーがカードを引く番になる
# - ディーラーは、カードの合計が17に達するまでカードを強制的に引く
# - ディーラーのカードの合計が21超えたら、プレイヤーの勝ち。ゲーム終了
# - ディーラーとプレイヤーとでカードの合計を比較して、高いほうが勝ち。ゲーム終了


# %%
from __future__ import absolute_import, division, print_function
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.dqn import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
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
    # Simplified Blackjack
    ACT_HIT = 0
    ACT_STICK = 1
    LIMIT_SCORE = 21
    STATE_LEN = 3

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, name='action',
            minimum=self.ACT_HIT, maximum=self.ACT_STICK,
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.STATE_LEN,), dtype=np.int32, minimum=0,
            name='observation'
        )
        self.reset()
        return

    def _state(self):
        # Full state includes 1st card of the dealer and all cards of player,
        # but this return only the last STATE_LEN cards.
        state = [self._dealer_cards[0]] + self._player_cards
        return np.array(state[-self.STATE_LEN:], dtype=np.int32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def __reset(self):
        self._player_cards = [self._new_card(), self._new_card()]
        self._dealer_cards = [self._new_card()]
        self._episode_ended = False

    def _reset(self):
        self.__reset()
        # self._current_time_step = time_step.restart(self._state())
        # return self._current_time_step
        return time_step.restart(self._state())

    def _new_card(self):
        # Simplified Blackjack rule
        new_card = np.random.randint(1, 11+1)
        return new_card

    def _dealer_hit(self):
        while np.sum(self._dealer_cards) < 17:
            self._dealer_cards.append(self._new_card())
        return np.sum(self._dealer_cards)

    def _player_score(self):
        return np.sum(self._player_cards)

    def _terminate(self, reward):
        print("Player: {} -> {}. Dealer: {} -> {}. Reward: {}.".format(
            self._player_cards, np.sum(self._player_cards),
            self._dealer_cards, np.sum(self._dealer_cards),
            reward))
        self._episode_ended = True
        return time_step.termination(self._state(), reward)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()  # don't forget to `return`

        if action == self.ACT_HIT:
            self._player_cards.append(self._new_card())
            if self._player_score() > self.LIMIT_SCORE:  # the player goes bust
                return self._terminate(-1)

            return time_step.transition(self._state(), reward=0, discount=1)

        # Afteward action == self.ACT_STICK
        dealer_score = self._dealer_hit()
        player_score = self._player_score()
        if dealer_score > self.LIMIT_SCORE or dealer_score < player_score:
            reward = 1
        elif dealer_score == player_score:
            reward = 0
        else:
            reward = -1
        return self._terminate(reward)


def print_spec(env):
    act_spec, ts_spec = env.action_spec(), env.time_step_spec()
    for x in (act_spec, ts_spec.observation, ts_spec.step_type,
              ts_spec.discount, ts_spec.reward):
        print(x)
    return

# TODO: validate_py_environment should check for a reset()
# utils.validate_py_environment(BlackJackEnv())

# %% [markdown]
# ## for loop でランダムに遊ぶ場合
#
# プレイヤーがカードを最大 `n_max_cards` 枚引く。
# 平均的に見たら負けています。
#
# %%


def play_blackjack(env, n_max_cards=1):
    ts = env.reset()
    gain = ts.reward
    cards = []
    for _ in range(np.random.randint(n_max_cards+1)):
        if ts.is_last():
            break
        ts = env.step(tf.constant([BlackJackEnv.ACT_HIT]))
        cards += [ts.observation[0][0].numpy()]
        gain += ts.reward

    if not ts.is_last():
        ts = env.step(tf.constant([BlackJackEnv.ACT_STICK]))
        gain += ts.reward
    gain = gain.numpy()[0]
    return cards, gain


env = tf_py_environment.TFPyEnvironment(BlackJackEnv())
gains = []
num_eval_episodes = 5  # @param
for _ in range(num_eval_episodes):
    _, gain = play_blackjack(env, 2)
    gains.append(gain)
mean_score1 = np.mean(gains)
mean_score1

# %% [markdown]
# ## RandomTFPolicyでランダムに遊ぶ
# %%
num_eval_episodes = 5  # @param


def evaluate_policy(
        policy,
        observers=None,
        eval_env=tf_py_environment.TFPyEnvironment(BlackJackEnv()),
        num_episodes=num_eval_episodes):
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env, policy, observers, num_episodes)
    final_step, policy_state = driver.run(num_episodes=num_episodes)
    print('Final step', final_step)
    return driver, final_step, policy_state


env = tf_py_environment.TFPyEnvironment(BlackJackEnv())
rand_policy = random_tf_policy.RandomTFPolicy(
    action_spec=env.action_spec(),
    time_step_spec=env.time_step_spec(),)
replay_buffer = []
avg_return = tf_metrics.AverageReturnMetric()
n_episodes = tf_metrics.NumberOfEpisodes()
n_steps = tf_metrics.EnvironmentSteps()
observers = [replay_buffer.append, avg_return, n_episodes, n_steps]
driver, final_step, policy_state = evaluate_policy(rand_policy, observers)
print('Number of Steps: ', n_steps.result().numpy())
print('Number of Episodes: ', n_episodes.result().numpy())
print('Average Return: ', avg_return.result().numpy())

# %%
final_step, policy_state = driver.run(final_step, policy_state, num_episodes=1)
print('Number of Steps: ', n_steps.result().numpy())
print('Number of Episodes: ', n_episodes.result().numpy())
print('Average Return: ', avg_return.result().numpy())

# %% [markdown]
# ## DQNで強化学習

# %%


learning_rate = 1e-3  # @param
replay_buffer_capacity = 1000  # @param
fc_layer_params = (100, 100)  # @param
num_iterations = 5000  # @param
initial_collect_steps = 1000  # @param
batch_size = 64  # @param
collect_steps_per_iteration = 10  # @param
log_interval = 200  # @param
eval_interval = 1000  # @param


class DqnAgent:
    def __init__(self, env):
        # Agent初期化
        self.env = env
        q_net = q_network.QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=fc_layer_params,
        )

        adam = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.compat.v2.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            env.time_step_spec(),
            env.action_spec(),
            q_network=q_net,
            optimizer=adam,
            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
            train_step_counter=train_step_counter,
        )
        self.agent.initialize()
        self._create_replay_buffer()

    def _create_replay_buffer(self):
        # Replay Bufferの初期化。初期データ収集
        self.replay_buffer = buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            # batch_size=self.env.batch_size,
            batch_size=batch_size,
            max_length=replay_buffer_capacity
        )
        print(buffer.capacity.numpy(), buffer._batch_size)
        print(buffer.data_spec)
        self._collect_data(rand_policy, initial_collect_steps)
        dataset = buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
        ).prefetch(3)
        self.data_iterator = iter(dataset)

    def _collect_data(self, policy, n_steps):
        # Replay Bufferへのデータ追加
        dynamic_step_driver.DynamicStepDriver(
            self.env, policy, [self.replay_buffer.add_batch], n_steps
        ).run()
        return

    def train(self, num_iterations=num_iterations):
        avg_return = evaluate_policy(self.agent.policy, num_eval_episodes)
        avg_returns = [avg_return]
        for step in range(1, 1 + num_iterations):
            self._collect_data(self.agent.collect_policy,
                               collect_steps_per_iteration)
            experience, _ = next(self.data_iterator)
            train_loss = self.agent.train(experience)
            if step % log_interval == 0:
                print('Step = {}, Loss= {}'.format(step, train_loss.loss))
            if step % eval_interval == 0:
                avg_return = evaluate_policy(
                    self.agent.policy, num_eval_episodes)
                print('Step = {}, Average Return= {}'.format(step, avg_return))
                avg_returns.append(avg_return)
        return avg_returns

env=BlackJackEnv()
dqn = DqnAgent(env)

# %%
# 以上
