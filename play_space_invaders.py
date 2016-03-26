#!/usr/bin/env python
#
import sys
import numpy as np
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
  print('Usage: %s rom_file' % sys.argv[0])
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt(b'random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
rom_file = str.encode(sys.argv[1])
ale.loadROM(rom_file)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

screen_width, screen_height = ale.getScreenDims();
screen_data = np.empty((screen_height, screen_width, 1), dtype=np.uint8)

# Play 10 episodes
for episode in range(10):
  total_reward = 0
  while not ale.game_over():
    a = legal_actions[randrange(len(legal_actions))]
    # Apply an action and get the resulting reward
    previous_lives = ale.lives()
    reward = ale.act(a);
    if ale.lives() < previous_lives or reward < 0:
        reward = -1
    elif reward > 1:
        reward = 1

    # Reward is now -1, 0, 1        
    #print('reward %d lives %d' % (reward, ale.lives()))
    total_reward += reward
    
    if (False and ale.getFrameNumber() % 100 == 0):
        ale.getScreenGrayscale(screen_data)
        print('screen:')
        for y in range(screen_height // 5):
            for x in range(screen_width // 2):
                pixel_sum = 0
                max_pixel = 0
                for yblock in range(5):
                    for xblock in range(2):
                        if screen_data[y * 5 + yblock, x * 2 + xblock] > max_pixel:
                            max_pixel = screen_data[y * 5 + yblock, x * 2 + xblock]
                        pixel_sum = np.int32(screen_data[y * 5 + yblock, x * 2 + xblock])
                pixel = pixel_sum // 10
                sys.stdout.write('#' if max_pixel > 50 else ' ')
            print('')
        print('')
              
  print('Episode %d ended with score: %d' % (episode, total_reward))
  ale.reset_game()
