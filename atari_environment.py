import blosc
import numpy as np
import scipy.ndimage as ndimage
import blosc
import os
import random
import zlib
from state import State
from ale_python_interface import ALEInterface

# Terminology in this class:
#   Episode: the span of one game life
#   Game: an ALE game (e.g. in space invaders == 3 Episodes or 3 Lives)
#   Frame: An ALE frame (e.g. 60 fps)
#   Step: An Environment step (e.g. covers 4 frames)
#
class AtariEnvironment:
    
    def __init__(self, args, outputDir):
        
        self.outputDir = outputDir
        self.screenCaptureFrequency = args.screen_capture_freq
        self.useCompression = args.compress_screens
        self.currentScreenBatch = None
        
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', 123456)
        random.seed(123456)
        # Fix https://groups.google.com/forum/#!topic/deep-q-learning/p4FAIaabwlo
        self.ale.setFloat(b'repeat_action_probability', 0.0)

        # Load the ROM file
        self.ale.loadROM(args.rom)

        self.actionSet = self.ale.getMinimalActionSet()
        self.gameNumber = 0
        self.stepNumber = 0
        self.resetGame()

    def getNumActions(self):
        return len(self.actionSet)

    def getState(self):
        return self.state
    
    def getGameNumber(self):
        return self.gameNumber
    
    def getFrameNumber(self):
        return self.ale.getFrameNumber()
    
    def getEpisodeFrameNumber(self):
        return self.ale.getEpisodeFrameNumber()
    
    def getEpisodeStepNumber(self):
        return self.episodeStepNumber
    
    def getStepNumber(self):
        return self.stepNumber
    
    def getGameScore(self):
        return self.gameScore

    def isGameOver(self):
        return self.ale.game_over()

    def step(self, action):
        previousLives = self.ale.lives()
        reward = 0
        isTerminal = 0
        self.stepNumber += 1
        self.episodeStepNumber += 1
        
        for i in range(4):
            prevScreenRGB = self.ale.getScreenRGB()
            reward += self.ale.act(self.actionSet[action])
            screenRGB = self.ale.getScreenRGB()
    
            # Detect end of episode, I don't think I'm handling this right in terms
            # of the overall game loop (??)
            if self.ale.lives() < previousLives or self.ale.game_over():
                isTerminal = 1
                break

            if self.gameNumber % self.screenCaptureFrequency == 0:
                dir = self.outputDir + '/screen_cap/game-%06d' % (self.gameNumber)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                self.ale.saveScreenPNG(dir + '/frame-%06d.png' % (self.getEpisodeFrameNumber()))

        maxedScreen = self._processFullScreen(np.maximum(screenRGB, prevScreenRGB))
        self.state = self.state.stateByAddingScreen(maxedScreen, self.ale.getFrameNumber())
        self.gameScore += reward
        return reward, self.state, isTerminal

    def resetGame(self):
        if self.ale.game_over():
            self.gameNumber += 1
        self.ale.reset_game()
        screen = self._processFullScreen(self.ale.getScreenRGB())
        self.state = State().stateByAddingScreen(screen, self.ale.getFrameNumber())
        self.gameScore = 0
        self.episodeStepNumber = 0 # environment steps vs ALE frames.  Will probably be 4*frame number

    def _processFullScreen(self, screen):
        screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        screen.resize((84, 84, 1))
        if self.useCompression:
            if self.currentScreenBatch is None or self.currentScreenBatch.isCompressed:
                self.currentScreenBatch = CompressedScreenBatch()
            return CompressedScreenReference(self.currentScreenBatch, screen)
        else:
            return SimpleScreenReference(screen)

class SimpleScreenReference:

    def __init__(self, screen):
        self.screen = screen

    def getPixels(self):
        return self.screen


class CompressedScreenBatch:
    batchSize = 10
    currentlyUncompressed = None

    def __init__(self):
        self.isCompressed = False
        self.screens = []
        self.compressed = None
        self.cache = None
    
    def addScreen(self, screen):
        if self.isCompressed:
            raise Exception('Cannot add a screen to a batch after it has been compressed')
        
        screenId = len(self.screens)
        self.screens.append(screen)
        
        if len(self.screens) == CompressedScreenBatch.batchSize:
            uncompressed = np.reshape(self.screens, (CompressedScreenBatch.batchSize, 84, 84)).tobytes()
            self.compressed = self._compress(uncompressed)
            self.isCompressed = True
            self.screens = None
            
        return screenId

    def getDecompressedScreen(self, screenId):
        if self.isCompressed:
            if self != CompressedScreenBatch.currentlyUncompressed:
                if CompressedScreenBatch.currentlyUncompressed != None:
                    CompressedScreenBatch.currentlyUncompressed.cache = None
                CompressedScreenBatch.currentlyUncompressed = self
                self.cache = np.reshape(np.fromstring(self._decompress(self.compressed), dtype=np.uint8), (CompressedScreenBatch.batchSize, 84, 84))
            screen = self.cache[screenId,...]
            screen.resize((84, 84, 1))
            return screen
        else:
            return self.screens[screenId]

    def _compress(self, data):
        #return blosc.compress(data, typesize=1)
        return zlib.compress(data, 9)

    def _decompress(self, data):
        #return blosc.decompress(data)
        return zlib.decompress(data)

class CompressedScreenReference:
    def __init__(self, batch, screen):
        self.batch = batch
        self.screenId = batch.addScreen(screen)

    def getPixels(self):
        screen = self.batch.getDecompressedScreen(self.screenId)
        return screen
