#!/usr/bin/env python
#
import sys
import numpy as np
import state as gs
import random
import os
import replay
import time
import argparse
import dqn
from ale_python_interface import ALEInterface

parser = argparse.ArgumentParser()
parser.add_argument("--replay-capacity", type=int, default=100000, help="how many states to store for future training")
parser.add_argument("--frame-sample-freq", type=int, default=4, help="how often to sample frames into the state")
parser.add_argument("--training-freq", type=int, default=4, help="how often (in frames) to train the network")
parser.add_argument("--screen-capture-freq", type=int, default=100, help="record screens for a game this often")
parser.add_argument("--observation-frames", type=int, default=5000, help="train only after this many frames")
parser.add_argument("--learning-rate", type=float, default=2e-4, help="learning rate (step size for optimization algo)")
parser.add_argument("rom", help="rom file to run")
args = parser.parse_args()

romFile = args.rom
replayMemoryCapacity = args.replay_capacity
frameSampleFrequency = args.frame_sample_freq
trainingFrequency = args.training_freq
minObservationFrames = args.observation_frames
screenCaptureFrequency = args.screen_capture_freq
learningRate = args.learning_rate

print 'Arguments: %s' % (args)

ale = ALEInterface()

# Get & Set the desired settings

ale.setInt(b'random_seed', 123456)
random.seed(123456)

# This is to deal with the issue mentioned in the paper for space invaders
# where bullets are only drawn on some frames
ale.setBool(b'color_averaging', True)

# Load the ROM file
ale.loadROM(romFile)

screenWidth, screenHeight = ale.getScreenDims();
screenData = np.empty((screenHeight, screenWidth, 1), dtype=np.uint8)
print('%d x %d' % (screenWidth, screenHeight))
actionSet = ale.getMinimalActionSet();

baseOutputDir = 'game-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(baseOutputDir)

dqn = dqn.DeepQNetwork(screenWidth, screenHeight, len(actionSet), baseOutputDir, learningRate)
replayMemory = replay.ReplayMemory(replayMemoryCapacity)
gameCount = 0


# Play 10 episodes
for episode in range(100000):
    
    gameScore = 0
    oldState = None
    ale.getScreenGrayscale(screenData)    
    state = gs.State().stateByAddingScreen(screenData)
    startTime = time.time()

    while not ale.game_over():
      
        action = dqn.chooseAction(state)

        # Apply an action and get the resulting reward
        previous_lives = ale.lives()
        reward = ale.act(actionSet[action])
        gameScore += reward
        
        # Convert reward to -1, 0, +1
        if ale.lives() < previous_lives or reward < 0:
            reward = -1
        elif reward > 1:
            reward = 1
        
        if ale.getFrameNumber() % frameSampleFrequency == 0:
            ale.getScreenGrayscale(screenData)    
            oldState = state
            state = state.stateByAddingScreen(screenData)
            replayMemory.addSample(replay.Sample(oldState, action, reward, state, ale.game_over()))

        if ale.getFrameNumber() > minObservationFrames and ale.getFrameNumber() % trainingFrequency == 0:
            # (??) batch size
            batch = replayMemory.drawBatch(32)
            dqn.train(batch)
        
        if gameCount % screenCaptureFrequency == 0:
            dir = baseOutputDir + '/screen_cap/game-%06d' % (gameCount)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            ale.saveScreenPNG(dir + '/frame-%06d.png' % (ale.getEpisodeFrameNumber()))

    episodeTime = time.time() - startTime
    print('Episode %d ended with score: %d (%d frames in %fs for %d fps)' % (episode, gameScore, ale.getEpisodeFrameNumber(), episodeTime, ale.getEpisodeFrameNumber() / episodeTime))
    ale.reset_game()
    gameCount += 1


def printScreen(screenData, screenWidth, screenHeight):
    print('screen:')
    for y in range(screenHeight // 5):
        for x in range(screenWidth // 2):
            pixelSum = 0
            maxPixel = 0
            for yblock in range(5):
                for xblock in range(2):
                    if screenData[y * 5 + yblock, x * 2 + xblock] > maxPixel:
                        max_pixel = screenData[y * 5 + yblock, x * 2 + xblock]
                    pixelSum = np.int32(screenData[y * 5 + yblock, x * 2 + xblock])
            pixel = pixelSum // 10
            sys.stdout.write('#' if maxPixel > 50 else ' ')
        print('')
    print('')
          
    