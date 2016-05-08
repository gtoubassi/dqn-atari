# DQN Atari

This repo represents my attempt to reproduce the DeepMind Atari playing agent described in the recent [Nature paper](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

While the DeepMind implementation is built in [lua with torch7](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner), this implementation uses [TensorFlow](http://tensorflow.org).  It also depends on the [Arcade Learning Environment](http://www.arcadelearningenvironment.org/).

### Results

I have been focused on attempting to match DeepMind's performance on Space Invaders (1976+/-800).  My current results are far short at ~900 (random agent scores ~150).  Thus far I have not found anyone that has reproduced the DeepMind results using the approach described in the Nature paper.  If you've done it, particularly with TensorFlow, let me know!

More results to follow...

### References

The following were very helpful:

* [Overview of Deep Q Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* [deep_rl_ale](https://github.com/Jabberwockyll/deep_rl_ale)
* [Flabbybird agent using TensorFlow](https://github.com/yenchenlin1994/DeepLearningFlappyBird)
* [Space Invaders using Theano](http://maciejjaskowski.github.io/2016/03/09/space-invaders.html)
* [Deep Q Learning Google Group](https://groups.google.com/forum/#!forum/deep-q-learning)

### Running on AWS

Details on how to get TensorFlow set up on AWS GPU instances can be found [here](https://github.com/gtoubassi/dqn-atari/wiki/Setting-up-TensorFlow-on-AWS-GPU)
