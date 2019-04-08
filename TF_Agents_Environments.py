# %% [markdown]
# # TF Agents の環境

# %%
# !pip install tf-nightly tf-agents-nightly 'gym==0.10.11'
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, suite_gym

tf.compat.v1.enable_v2_behavior()

# %% [markdown]
# ## Python環境
# %%
# Python Environment
py_env = suite_gym.load("CartPole-v0")
act_spec, ts_spec = py_env.action_spec(), py_env.time_step_spec()
for x in (act_spec, ts_spec.observation, ts_spec.step_type, ts_spec.discount, ts_spec.reward):
    print(x)

ts = py_env.reset()
gain = 0
while not ts.is_last():
    action = np.random.randint(2)
    ts = py_env.step(action)
    print('.', end='')
    gain += ts.reward
print("\nTotal gain:", gain)


# %% [markdown]
# ## Python環境をTF環境でラッピング
# %%
# Wrapping a PyEnv in TF
tf_env = tf_py_environment.TFPyEnvironment(py_env)
act_spec, ts_spec = tf_env.action_spec(), tf_env.time_step_spec()
for x in (act_spec, ts_spec.observation, ts_spec.step_type, ts_spec.discount, ts_spec.reward):
    print(x)

ts = tf_env.reset()
gain = 0
while not ts.is_last():
    action = tf.random_uniform([1], 0, 2, dtype=tf.int32)
    ts = tf_env.step(action)
    print('.', end='')
    gain += ts.reward
print("\n", gain)
print("\nTotal gain:", gain.numpy()[0])
