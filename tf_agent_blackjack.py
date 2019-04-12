#%% [markdown]
#  # TF Agent で Blackjack 遊ぶ
#
#  ブラックジャックを参考に、次のような
#  ルールとする。
#
#  ## ルール
#
#  - カードの値は、 1〜10の間にランダムに決まる（エースは1か11）
#  - 最初にカードをプレイヤーに2枚、ディーラーに1枚
#  - プレイヤーが何枚でもカードを引ける(hit)が、合計が21超えたら即負け。ゲーム終了
#  - プレイヤーがカードを引くのを止めたら(stick)、ディーラーがカードを引く番になる
#  - ディーラーは、カードの合計が17に達するまでカードを強制的に引く
#  - ディーラーのカードの合計が21超えたら、プレイヤーの勝ち。ゲーム終了
#  - ディーラーとプレイヤーとでカードの合計を比較して、プレイヤーが高いなら勝ち。引き分けはなし。ゲーム終了
#  - プレイヤーが勝つ場合 1 点。負け（引き分けも）は 0 点
#
# ## 参考
#
# - [TF Agents DQN example](https://github.com/tensorflow/agents/blob/154b81176041071a84b72eb64d419d256dcc947a/tf_agents/agents/dqn/examples/v2/train_eval.py)
# - [Kaggle BlackJack env](https://github.com/Kaggle/learntools/blob/master/learntools/python/blackjack.py)
# - [How to match DeepMind’s Deep Q-Learning score in Breakout](https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756)

#%%
# !which python
# !sudo apt install -y cuda-cublas-10-0  cuda-cusolver-10-0 cuda-cudart-10-0 cuda-cusparse-10-0
# !conda install -y -c anaconda cudatoolkit
# !pip install --upgrade tf-nightly-gpu tf-agents-nightly gym
# !pip install --upgrade tensorflow-gpu==2.0.0-alpha0 tf-agents-nightly gym
# !pip install --upgrade tf-nightly-gpu tf-agents-nightly gym


#%%
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents.dqn import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
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
assert tf.executing_eagerly()
# tf.enable_eager_execution()


DEBUG = 1 # @param
WIN_SCORE = 1 # @param
LOSS_SCORE = DRAW_SCORE = 0 # @param


def plog(msg, *args):
    if not DEBUG: return
    if len(args) == 0:
        print(msg)
    else:
        print(msg.format(*args))

#%% [markdown]
# ## ブラックジャック環境定義

#%%
class CardSet:
    LIMIT_SCORE = 21
    DEALER_MIN = 17
    ACE_VAL = 1
    def __init__(self, is_dealer=False):
        self.cards = []
        if is_dealer:
            self.dealer_hit()
        else:
            self.hit(), self.hit()

    def add(self, card):
        self.cards.append(card)

    def sum(self):
        sum = np.sum(self.cards)

        for i in range(1, self.aces()+1):
            if (sum + 10) <= self.LIMIT_SCORE:
                sum += 10
            else:
                break
        return sum

    def aces(self):
        n_aces = len(self.cards) - np.count_nonzero(np.array(self.cards)-
                                                   self.ACE_VAL)
        # print(f'n_aces {n_aces}')
        return n_aces

    def is_bust(self):
        return self.sum() > self.LIMIT_SCORE

    def __getitem__(self, key):
        return self.cards[key]

    def hit(self):
        # Simplified Blackjack rule
        new_card = np.random.randint(1, 10+1)
        self.add(new_card)

    def dealer_hit(self):
        while self.sum() < self.DEALER_MIN:
            self.hit()
        return self.sum()

    def _player_score(self):
        return np.sum(self._player_cards)

    def __str__(self):
        return str(self.cards)


class BlackJackEnv(py_environment.PyEnvironment):
    # Simplified Blackjack
    ACT_HIT = 0
    ACT_STICK = 1
    STATE_LEN = 3

    def __init__(self):
        self._batch_size = 1  # batch_size
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
        state = [self._dealer_cards[0],
                 self._player_cards.sum(), self._player_cards.aces()]
        assert len(state) == self.STATE_LEN
        return np.array(state, dtype=np.int32)

    def _state_last_cards(self):
        # Full state includes 1st card of the dealer and all cards of player,
        # but this return only the last _state_len cards.
        state = [self._dealer_cards[0]] + self._player_cards.cards
        if len(state) < self._state_len:
            state = np.pad(state, (0, self._state_len-len(state)),
                           'constant', constant_values=(0))
        return np.array(state[-self._state_len:], dtype=np.int32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def __reset(self):
        self._player_cards = CardSet(is_dealer=False)
        self._dealer_cards = CardSet(is_dealer=True)
        self._episode_ended = False

    def _reset(self):
        self.__reset()
        return time_step.restart(self._state())

    def _terminate(self, reward):
        plog(
            "Player: {} -> {}. Dealer: {} -> {}. Reward: {}.",
            self._player_cards, self._player_cards.sum(),
            self._dealer_cards, self._dealer_cards.sum(),
            reward)
        self._episode_ended = True
        return time_step.termination(self._state(), reward)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()  # don't forget to `return`

        if action == self.ACT_HIT:
            self._player_cards.hit()
            if self._player_cards.is_bust():
                return self._terminate(LOSS_SCORE)

            return time_step.transition(self._state(), reward=0, discount=1)

        # Afteward action == self.ACT_STICK
        dealer_score = self._dealer_cards.dealer_hit()
        player_score = self._player_cards.sum()
        if self._dealer_cards.is_bust() or dealer_score < player_score:
            reward = WIN_SCORE
        else:
            reward = LOSS_SCORE
        return self._terminate(reward)


    @classmethod
    def tf_env(cls):
        return tf_py_environment.TFPyEnvironment(cls())


def print_spec(env):
    act_spec, ts_spec = env.action_spec(), env.time_step_spec()
    for x in (act_spec, ts_spec.observation, ts_spec.step_type,
              ts_spec.discount, ts_spec.reward):
        print(x)
    return


# TODO: validate_py_environment should check for a reset()
utils.validate_py_environment(BlackJackEnv())

#%% [markdown]
# ## 既知の戦略で勝率<38%
#
# - [Kaggle BlackJack env](https://github.com/Kaggle/learntools/blob/master/learntools/python/blackjack.py)
# - [Kaggle BlackJack microchallange forum](https://www.kaggle.com/learn-forum/58735#latest-348767)

#%%
def should_hit(dealer_card_val, player_total, player_aces):
    """既知の戦略
    Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay. player_aces is the number of aces the player has.
    Strategy learnt from https://www.kaggle.com/learn-forum/58735#latest-348767
    """
    hit = (
        (player_total <= 11) or
        (player_total == 12 and (dealer_card_val < 4 or dealer_card_val > 6)) or
        (player_total <= 16 and (dealer_card_val > 6)) or
        (player_total == 17 and (dealer_card_val == 1))
    )
#     print(f"Dealer {dealer_card_val}. Player {player_total}, aces {player_aces}, hit {hit}")
    return hit

def play_known_strategy(env):
    ts = env.reset()
    gain = ts.reward
    dealer_card_val, player_total, player_aces = ts.observation.numpy()[0]
    while should_hit(dealer_card_val, player_total, player_aces):
        ts = env.step(tf.constant([BlackJackEnv.ACT_HIT]))
        gain += ts.reward
        dealer_card_val, player_total, player_aces = ts.observation.numpy()[0]
    if not ts.is_last():
        ts = env.step(tf.constant([BlackJackEnv.ACT_STICK]))
        gain += ts.reward
    gain = gain.numpy()[0]
    return gain

DEBUG = 0  # @param
log_interval = 1000  # @param
num_iterations = 10_000  # @param

env = BlackJackEnv.tf_env()
gains = []
avg_gains =[]
for step in range(1, 1+num_iterations):
    gain = play_known_strategy(env)
    gains.append(gain)
    if step % log_interval == 0:
        avg_gain = np.mean(gains)
        avg_gains.append(avg_gain)
        print(f'Step {step: >3}. Gain {avg_gain}.')
plt.plot(avg_gains)

#%% [markdown]
#  ## デタラメに遊ぶ場合
#
#  プレイヤーがカードを最大 `n_max_cards` 枚引く。
#  平均的に見たら負けています。
#
#%% [markdown]
# ### 最大１枚引く場合、勝率<30.6%

#%%
def play_blackjack(env, n_max_cards=1):
    ts = env.reset()
    gain = ts.reward
    for _ in range(np.random.randint(n_max_cards+1)):
        if ts.is_last():
            break
        ts = env.step(tf.constant([BlackJackEnv.ACT_HIT]))
        assert ts.reward >= LOSS_SCORE
        gain += ts.reward

    if not ts.is_last():
        ts = env.step(tf.constant([BlackJackEnv.ACT_STICK]))
        gain += ts.reward
    gain = gain.numpy()[0]
    return gain


#%%
DEBUG = 0  # @param
num_iterations = 5_000  # @param
log_interval = 1_000  # @param
n_max_cards = 1  # @param

env = BlackJackEnv.tf_env()
gains = []
for step in range(1, 1+num_iterations):
    gain = play_blackjack(env, n_max_cards)
    gains.append(gain)
    if step % log_interval == 0:
        print(f'Step {step: >3}. Gain {np.mean(gains)}.')

#%% [markdown]
# ### 最大2枚引く場合、勝率<28.7%

#%%
num_iterations = 5_000  # @param
log_interval = 1_000  # @param
n_max_cards = 2  # @param

env = BlackJackEnv.tf_env()
gains = []
for step in range(1, 1+num_iterations):
    gain = play_blackjack(env, n_max_cards)
    gains.append(gain)
    if step % log_interval == 0:
        print(f'Step {step: >3}. Gain {np.mean(gains)}.')

#%% [markdown]
#  ### RandomTFPolicyで勝率<28.3%
#
# `tf_metrics.AverageReturnMetric(buffer_size)` を使う時には buffer_size に勝率を図るエピソード数を代入。

#%%
class PolicyEvaluator:
    def __init__(self,
        eval_env,
        n_eval_episodes
    ):
        self.eval_env = eval_env
        avg_return = tf_metrics.AverageReturnMetric(buffer_size=n_eval_episodes)
        n_episodes = tf_metrics.NumberOfEpisodes()
        self.observers = [avg_return, n_episodes, ]
        self.n_eval_episodes = n_eval_episodes

    def evaluate_policy(self, policy):
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.eval_env, policy, self.observers, self.n_eval_episodes)
        driver.run()
        avg_return, n_episodes = self.observers
        avg_return = avg_return.result().numpy()
        n_episodes = n_episodes.result().numpy()
        n_win = int(n_episodes * avg_return)
        print(f'Evaluated n_episodes {n_episodes: >3}. avg_return {avg_return:f}. n_win {n_win}')
        return avg_return

def repeat_evaluate_random_policy(n_iterations, n_episodes):
    env = BlackJackEnv.tf_env()
    policy = random_tf_policy.RandomTFPolicy(
        action_spec=env.action_spec(),
        time_step_spec=env.time_step_spec(),)
    evaluator = PolicyEvaluator(env, n_episodes)
    gains = []
    for step in range(1, 1+n_iterations//n_episodes):
        avg_return = evaluator.evaluate_policy(policy)
        gains.append(avg_return)
    return gains

gains = repeat_evaluate_random_policy(5_000, 1000)
plt.plot(gains)

#%% [markdown]
#  ## DQN強化学習
#
# Agentに見せる環境の情報 (state) のパターン
# - ディーラーの最初のカード
# - プレイヤーのカードの合計値
# - プレイヤーのエースカードの数
#

#%%
class DqnAgent:
    def __init__(self, env):
        # Agent初期化
        self.env = env
        q_net = q_network.QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params=fc_layer_params,
        )

        adam = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=0.8, epsilon=1)

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
        eval_env = BlackJackEnv.tf_env()
        self.evaluator = PolicyEvaluator(eval_env, n_eval_episodes)

    # TODO: try different num_steps value
    def _create_replay_buffer(self, num_steps=2):
        # Replay Bufferの初期化。初期データ収集
        self.replay_buffer = buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,  # actually 1, env isn't batched
            max_length=replay_buffer_capacity
        )
        print(buffer.capacity.numpy(), buffer._batch_size)
        print(buffer.data_spec)
        self._collect_data(
            self.agent.collect_policy,
            initial_collect_steps)
        dataset = buffer.as_dataset(
            num_parallel_calls=3, num_steps=num_steps,
            sample_batch_size=batch_size,
        ).prefetch(batch_size)
        self.data_iterator = iter(dataset)

    def _collect_data(self, policy, n_steps):
        # Replay Bufferへのデータ追加
        dynamic_step_driver.DynamicStepDriver(
            self.env, policy, [self.replay_buffer.add_batch], n_steps
        ).run()
        return

    def train(self, num_iterations):
        avg_returns = []
        for step in range(1, 1 + num_iterations):
            self._collect_data(self.agent.collect_policy,
                               collect_steps_per_iteration)
            experience, _ = next(self.data_iterator)
            train_loss = self.agent.train(experience)
            self._print_log(step, train_loss.loss, avg_returns)
        return avg_returns

    def _print_log(self, step, loss, avg_returns):
        if step % log_interval == 0:
            print(f'Step {step: >3}. Loss {loss}.')
        if step % eval_interval == 0:
            avg_return = self.evaluator.evaluate_policy(self.agent.policy)
            print(f'Step {step: >3}. AvgReturn {avg_return}.')
            avg_returns.append(avg_return)

def plot(avg_returns, num_iterations, eval_interval):
    steps = range(0, num_iterations + 1, eval_interval)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.plot(steps, avg_returns)
    # plt.ylim(top=210)

#%% [markdown]
# ### 勝率36.5%

#%%
DEBUG = False
log_interval = 100  # @param
eval_interval = 500  # @param
num_iterations = 100_000  # @param
learning_rate = 5e-6  # @param
batch_size = 1000  # @param
collect_steps_per_iteration = 8  # @param
initial_collect_steps = batch_size  # @param
n_eval_episodes = 1000  # @param
replay_buffer_capacity = 10_000  # @param
fc_layer_params = (100, )  # @param


dqn = DqnAgent(BlackJackEnv.tf_env())
avg_returns = dqn.train(num_iterations)

plot(avg_returns, num_iterations, eval_interval)
