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
parser.add_argument("--replay-capacity", type=int, default=1000000, help="how many states to store for future training")
parser.add_argument("--frame-sample-freq", type=int, default=4, help="how often to sample frames into the state")
parser.add_argument("--training-freq", type=int, default=4, help="how often (in frames) to train the network")
parser.add_argument("--screen-capture-freq", type=int, default=100, help="record screens for a game this often")
parser.add_argument("--save-model-freq", type=int, default=1000, help="save the model once per 1000 training sessions")
parser.add_argument("--observation-frames", type=int, default=50000, help="train only after this many frames")
parser.add_argument("--learning-rate", type=float, default=0.00025, help="learning rate (step size for optimization algo)")
parser.add_argument("--target-model-update-freq", type=int, default=10000, help="how often to snapshot the model to update the target network during training (per nature paper)")
parser.add_argument("--model", help="tensorflow model checkpoint file to initialize from")
parser.add_argument("--eval-epsilon", type=float, help="If set, model will run without training, purely in eval mode, with the specified epsilon")
parser.add_argument("rom", help="rom file to run")
args = parser.parse_args()

romFile = args.rom
replayMemoryCapacity = args.replay_capacity
frameSampleFrequency = args.frame_sample_freq
trainingFrequency = args.training_freq
minObservationFrames = args.observation_frames
screenCaptureFrequency = args.screen_capture_freq
saveModelFrequency = args.save_model_freq
targetModelUpdateFrequency = args.target_model_update_freq
learningRate = args.learning_rate
modelFile = args.model
evalEpsilon = args.eval_epsilon

print 'Arguments: %s' % (args)

if evalEpsilon is None:
    print 'TRAINING MODE'
else:
    print 'EVAL MODE'

ale = ALEInterface()

# Get & Set the desired settings

ale.setInt(b'random_seed', 123456)
random.seed(123456)

# Fix https://groups.google.com/forum/#!topic/deep-q-learning/p4FAIaabwlo
ale.setFloat(b'repeat_action_probability', 0.0)

# Load the ROM file
ale.loadROM(romFile)

screenWidth, screenHeight = ale.getScreenDims();
print('%d x %d' % (screenWidth, screenHeight))
actionSet = ale.getMinimalActionSet();

baseOutputDir = 'game-out-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(baseOutputDir)

dqn = dqn.DeepQNetwork(screenWidth, screenHeight, len(actionSet), baseOutputDir, learningRate, modelFile, saveModelFrequency, targetModelUpdateFrequency, evalEpsilon)
replayMemory = replay.ReplayMemory(replayMemoryCapacity)

# Train for ~ 250,000 frames (to the nearest game).  This is probably ~30min    
def trainEpoch():
    episode = 0
    frameStart = ale.getFrameNumber()
    while ale.getFrameNumber() - frameStart < 250000:
    
        gameScore = 0
        oldState = None
        state = gs.State().stateByAddingScreen(ale.getScreenRGB(), ale.getFrameNumber())
        startTime = lastLogTime = time.time()
        lastRgbScreen = None
        stateReward = 0
        isTerminal = 0

        while not ale.game_over():
      
            action, futureReward = dqn.chooseAction(state)

            # Apply an action and get the resulting reward
            previous_lives = ale.lives()
            reward = ale.act(actionSet[action])
            gameScore += reward
            stateReward += reward
        
            # Detect end of episode, I don't think I'm handling this right in terms
            # of the overall game loop (??)
            if ale.lives() < previous_lives or ale.game_over():
                isTerminal = 1
        
            rgbScreen = ale.getScreenRGB()
            if evalEpsilon is None and (isTerminal or ale.getFrameNumber() % frameSampleFrequency == 0):
                maxedScreen = np.maximum(rgbScreen, lastRgbScreen) if lastRgbScreen is not None else rgbScreen
                oldState = state
                state = state.stateByAddingScreen(maxedScreen, ale.getFrameNumber())
                clippedReward = min(1, max(-1, stateReward))
                replayMemory.addSample(replay.Sample(oldState, action, clippedReward, state, isTerminal))
                stateReward = 0
                
            lastRgbScreen = rgbScreen

            if time.time() - lastLogTime > 60:
                print('  ...frame %d' % ale.getEpisodeFrameNumber())
                lastLogTime = time.time()

            if evalEpsilon is None and ale.getFrameNumber() > minObservationFrames and (isTerminal or ale.getFrameNumber() % trainingFrequency == 0):
                # (??) batch size
                batch = replayMemory.drawBatch(32)
                dqn.train(batch)

            if isTerminal:
                isTerminal = 0
                oldState = None
                state = gs.State().stateByAddingScreen(ale.getScreenRGB(), ale.getFrameNumber())
        
            if episode % screenCaptureFrequency == 0:
                dir = baseOutputDir + '/screen_cap/game-%06d' % (episode)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                ale.saveScreenPNG(dir + '/frame-%06d.png' % (ale.getEpisodeFrameNumber()))

        episodeTime = time.time() - startTime
        print('Episode %d ended with score: %d (%d frames in %fs for %d fps)' % (episode, gameScore, ale.getEpisodeFrameNumber(), episodeTime, ale.getEpisodeFrameNumber() / episodeTime))
        ale.reset_game()
        episode += 1


# Eval for ~ 125,000 frames (to the nearest game).  This is probably ~2-3min    
def evalEpoch():
    episode = 0
    frameStart = ale.getFrameNumber()
    totalScore = 0
    while ale.getFrameNumber() - frameStart < 60000:
    
        startTime = lastLogTime = time.time()

        while not ale.game_over():
      
            action, futureReward = dqn.chooseAction(state, overrideEpsilon=.05)

            # Apply an action and get the resulting reward
            totalScore += ale.act(actionSet[action])

        ale.reset_game()
        episode += 1
    print('Eval performance: %f' % (totalScore / episode))


while True:
    trainEpoch()
    evalEpoch()
