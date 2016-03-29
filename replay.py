import random

class Sample:
    
    def __init__(self, state1, action, reward, state2, terminal):
        self.state1 = state1
        self.action = action
        self.reward = reward
        self.state2 = state2
        self.terminal = terminal


class ReplayMemory:
    
    def __init__(self, maxSamples):
        self.samples = []
        self.maxSamples = maxSamples

    def numSamples():
        return len(self.samples)

    # ():) Per dqn paper only keep last N samples for memory purposes
    def addSample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.maxSamples * 1.05:
            len_before = len(self.samples)
            self.samples = self.samples[(len(self.samples) - self.maxSamples):]
    
    def drawBatch(self, batchSize):
        if batchSize > len(self.samples):
            raise IndexError('Too few samples (%d) to draw a batch of %d' % (len(self.samples), batchSize))
        return random.sample(self.samples, batchSize)
            
