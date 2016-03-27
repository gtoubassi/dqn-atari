#!/usr/bin/env python
#
import sys
import numpy as np
import state as gs
import random
import os
import replay
import time
import dqn
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print('Usage: %s rom_file' % sys.argv[0])
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings

ale.setInt(b'random_seed', 123456)
random.seed(123456)

# This is to deal with the issue mentioned in the paper for space invaders
# where bullets are only drawn on some frames (?? is it a net win)
ale.setBool(b'color_averaging', True)

# Load the ROM file
rom_file = str.encode(sys.argv[1])
ale.loadROM(rom_file)

screenWidth, screenHeight = ale.getScreenDims();
screenData = np.empty((screenHeight, screenWidth, 1), dtype=np.uint8)
print('%d x %d' % (screenWidth, screenHeight))
actionSet = ale.getMinimalActionSet();
dqn = dqn.DeepQNetwork(screenWidth, screenHeight, actionSet) # (??) Can replace actionSet here with len(actionSet)
replayMemory = replay.ReplayMemory(20000)
trainingFrequency = 4 # train every 4 frames
minObservationFrames = 1000
screenCaptureFrequency = 50
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
        
        ale.getScreenGrayscale(screenData)    
        oldState = state
        state = state.stateByAddingScreen(screenData)
        replayMemory.addSample(replay.Sample(oldState, action, reward, state, ale.game_over()))
        
        # Train every trainingFrequency steps
        if ale.getFrameNumber() > minObservationFrames and ale.getFrameNumber() % trainingFrequency == 0:
            # (??) batch size
            batch = replayMemory.drawBatch(32)
            dqn.train(batch)
        
        if gameCount % screenCaptureFrequency == 0:
            dir = 'screen_cap/game-%06d' % (gameCount)
            if ale.getEpisodeFrameNumber() == 1:
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
          
    