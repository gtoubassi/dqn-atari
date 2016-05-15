# DQN Atari

This repo represents my attempt to reproduce the DeepMind Atari playing agent described in the recent [Nature paper](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

While the DeepMind implementation is built in [lua with torch7](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner), this implementation uses [TensorFlow](http://tensorflow.org).  Like DeepMind, tt also depends on the [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) (technically I believe DeepMind uses their [Xitari](https://github.com/deepmind/xitari) fork of ALE).

### Results

I have been focused on attempting to match DeepMind's performance on Space Invaders (1976+/-800).  My current results are far short at ~1100 (random agent scores ~150).  Thus far I have not found anyone that has reproduced the DeepMind results using the approach described in the Nature paper.  If you've done it, particularly with TensorFlow, let me know!

I have also tried breakout and got a score of 130 but that was an older version with many minor issues (DeepMind reported 400+/-30).

A publicly viewable google spreadsheet has [results](https://docs.google.com/spreadsheets/d/1RZM2qhKQaXaud4S2ILsRVukmiPCjM-xtJTuPRpb96HY/edit#gid=2001383367) for various experiments I have run.

### Running

1. Get Python and Tensorflow running, preferably on a GPU (see notes on [AWS setup](https://github.com/gtoubassi/dqn-atari/wiki/Setting-up-TensorFlow-on-AWS-GPU)).
2. Install the arcade learning environment (see [wiki](https://github.com/gtoubassi/dqn-atari/wiki/Installing-ALE))
3. Download a game rom, and name it properly like space_invaders.bin (all lower case ending in bin -- the names must match for ALE).
4. Get the repo:

        git clone https://github.com/gtoubassi/dqn-atari.git

5. Run it!  The default parameters attempt to mimic the Nature paper configuration:

        cd dqn-atari
	    python ./play_atari.py ~/space_invaders.bin | tee train.log

6. Periodically check progress

        ./logstats.sh train.log

### References

The following were very helpful:

* [Overview of Deep Q Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
* David Silver's [Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa)
* [deep_rl_ale](https://github.com/Jabberwockyll/deep_rl_ale)
* [Flabbybird agent using TensorFlow](https://github.com/yenchenlin1994/DeepLearningFlappyBird)
* [Space Invaders using Theano](http://maciejjaskowski.github.io/2016/03/09/space-invaders.html)
* [Deep Q Learning Google Group](https://groups.google.com/forum/#!forum/deep-q-learning)
