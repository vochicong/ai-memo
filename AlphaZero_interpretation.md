# AlphaZero 解説
> *Draft, Apr. 5th, 2019*

What exactly the Alphas do?
Why is AlphaZero stronger than AlphaGo Zero?
Can we use/enjoy Alpha algorithms on a general CPU/GPU
or even a mobile device?

Alpha variants

- AlphaGo, 2016
- AlphaGo Zero, 2017a
- AlphaZero, 2017b

AlphaGo Zero is stronger than AlphaGo, since it has discovered some superhuman strategy for playing Go through self-plays. However it is not clear whether AlphaZero is ultimately stronger than AlphaGo Zero.

## AlphaGo

Training data for SL (supervised learning) is a collection of nearly 30 milion human expert moves.

"Asynchronouse policy and value MCTS", or APV-MCTS expands its tree by choosing an action according to probabilities supplied by a 13-layer deep convolutional ANN, called the *SL policy network*.

Value of a newly-added node $s$ is a mixing

$$ v(s) = (1-\eta)v_\theta(s) + \eta G$$

where
$v_\theta$ is a value function learned by a RL method, and
$G$ is the return of the **rollout** from state $s$.

The **rollout** policy is a fast, simple linear network trained by SL.

![AlphaGo pipeline](https://www.researchgate.net/profile/Daniele_Grattarola/publication/323218981/figure/fig15/AS:594583629090816@1518771192934/Neural-network-training-pipeline-of-AlphaGo-image-taken-from-39.png)

The *RL policy network* has the same structure as the SL policy network. It is initialized with the final weights of the SL policy network, and improved through policy-gradient reinforcement learning.

The *value network* has the same structure as the SL (and RL) policy network, and is trained by Monte Carlo policy evaluation on a large number of self-played games with moves selected by the RL policy network.


## AlphaZero

AlphaZero is simpler and more general then AlphaGo Zero.

The NN (neural network) $f_\theta$ starts from randomly initialized parameters $\theta$. It evaluates a specific game position $s$,

$$ ({\bf p}, v) = f_\theta(s), $$

where ${\bf p}$ is the policy vector and $v$ is the predicted outcome.

An MCTS (Monte Carlo Tree Search) upon a position
$s_{root} = s_t$ at turn $t$ selects a move
$a_t \sim {\bf \pi_t}$
where the search probabilities $\bf \pi_t$ relates to the visit counts at the state.

Loss function is a sum over a set of self-play games,

$$ l_\theta = \sum\left\{
(z-v)^2 - {\bf \pi}^\top\log{\bf p}
\right\}
+c||\theta||^2$$

where $c$ is a $L_2$ weight regularization parameter.

## AlphaGo Zero


---
## Refrences

- Sutton & Barto Book: Reinforcement Learning: An Introduction, 2nd ed.
- AlphaGo, AlphaGo Zero and AlphaZero related papers by DeepMind
