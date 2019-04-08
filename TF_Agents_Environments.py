# %%
!pip install tf-nightly tf-agents-nightly 'gym==0.10.11'
# %%
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, suite_gym

tf.compat.v1.enable_v2_behavior()
#%%
# A Standard Environment
env = suite_gym.load("CartPole-v0")
act_spec, ts_spec = env.action_spec(), env.time_step_spec()
act_spec, ts_spec.observation, ts_spec.step_type, ts_spec.discount, ts_spec.reward
