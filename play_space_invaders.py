#!/usr/bin/env python
#
import sys
import numpy as np
import os
import replay
import time
import argparse
import dqn
from atari_environment import AtariEnvironment

parser = argparse.ArgumentParser()
parser.add_argument("--train-epoch-frames", type=int, default=250000, help="how many frames to run during a training epoch (approx -- will finish current game)")
parser.add_argument("--eval-epoch-frames", type=int, default=100000, help="how many frames to run during an eval epoch (approx -- will finish current game)")
parser.add_argument("--replay-capacity", type=int, default=1000000, help="how many states to store for future training")
parser.add_argument("--screen-capture-freq", type=int, default=100, help="record screens for a game this often")
parser.add_argument("--save-model-freq", type=int, default=1000, help="save the model once per 1000 training sessions")
parser.add_argument("--observation-frames", type=int, default=50000, help="train only after this many frames")
parser.add_argument("--learning-rate", type=float, default=0.00025, help="learning rate (step size for optimization algo)")
parser.add_argument("--target-model-update-freq", type=int, default=10000, help="how often to snapshot the model to update the target network during training (per nature paper)")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
parser.add_argument("rom", help="rom file to run")
args = parser.parse_args()

trainEpochFrames = args.train_epoch_frames
evalEpochFrames = args.eval_epoch_frames
replayMemoryCapacity = args.replay_capacity
minObservationFrames = args.observation_frames

print 'Arguments: %s' % (args)

baseOutputDir = 'game-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(baseOutputDir)

environment = AtariEnvironment(args, baseOutputDir)

dqn = dqn.DeepQNetwork(environment.getNumActions(), baseOutputDir, args)
replayMemory = replay.ReplayMemory(replayMemoryCapacity)

def runEpoch(minEpochFrames, evalWithEpsilon=None):
    frameStart = environment.getFrameNumber()
    isTraining = True if evalWithEpsilon is None else False
    startGameCount = environment.getGameCount()
    epochTotalScore = 0

    while environment.getFrameNumber() - frameStart < minEpochFrames:
    
        startTime = lastLogTime = time.time()
        stateReward = 0
        state = None

        while not environment.isGameOver():
      
            action, futureReward = dqn.chooseAction(environment.getState(), overrideEpsilon=evalWithEpsilon)

            oldState = state
            reward, state, isTerminal = environment.step(action)

            if isTraining and oldState is not None:
                clippedReward = min(1, max(-1, reward))
                replayMemory.addSample(replay.Sample(oldState, action, clippedReward, state, isTerminal))
                if environment.getFrameNumber() > minObservationFrames:
                    batch = replayMemory.drawBatch(32)
                    dqn.train(batch)
        
            if time.time() - lastLogTime > 60:
                print('  ...frame %d' % environment.getEpisodeFrameNumber())
                lastLogTime = time.time()

            if isTerminal:
                state = None

        episodeTime = time.time() - startTime
        print('%s %d ended with score: %d (%d frames in %fs for %d fps)' % ('Episode' if isTraining else 'Eval', environment.getGameCount(), environment.getGameScore(), environment.getEpisodeFrameNumber(), episodeTime, environment.getEpisodeFrameNumber() / episodeTime))
        environment.resetGame()
        epochTotalScore += environment.getGameScore()
    return epochTotalScore / (environment.getGameCount() - startGameCount)
    


while True:
    aveScore = runEpoch(trainEpochFrames) #train
    print('Average training score: %d' % (aveScore))
    aveScore = runEpoch(evalEpochFrames, evalWithEpsilon=.05) #eval
    print('Average eval score: %d' % (aveScore))
