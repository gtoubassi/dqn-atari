import random
import bisect

class Sample:
    
    def __init__(self, state1, action, reward, state2, terminal):
        self.state1 = state1
        self.action = action
        self.reward = reward
        self.state2 = state2
        self.terminal = terminal
        self.weight = 1.0
        self.cumulativeWeight = 0
    
    def isInteresting(self):
        return self.reward != 0 or self.terminal
        
    def __cmp__(self, other):
        return cmp(self.cumulativeWeight, other.cumulativeWeight)


class ReplayMemory:
    
    def __init__(self, maxSamples):
        self.samples = []
        self.maxSamples = maxSamples
        self.interestingSampleRatio = 1

    def numSamples():
        return len(self.samples)

    # ():) Per dqn paper only keep last N samples for memory purposes
    def addSample(self, sample):

        sampleSize = 50000
        updateFreq = 1000
        if len(self.samples) > sampleSize and len(self.samples) % updateFreq == 0:
            numInteresting = 1
            for i in xrange(len(self.samples) - 1, len(self.samples) - sampleSize - 1, -1):
                if self.samples[i].isInteresting():
                    numInteresting += 1
            self.interestingSampleRatio = float(numInteresting) / sampleSize


        if sample.isInteresting():
            sample.weight *= 1000

        if len(self.samples) == 0:
            sample.cumulativeWeight = sample.weight
        else:
            sample.cumulativeWeight = sample.weight + self.samples[-1].cumulativeWeight
        self.samples.append(sample)
        
#        if sample.isInteresting():
#            interestingSampleWeightMultiplier = 20
#            
#            howManySamplesToBias = max(1, int(1.0/(2 * float(self.interestingSampleRatio))))
            # If howManySamplesToBias is 10, then we want to bias the most recent samples's weight by 2, and the 9th most recent sample by 1.1
            # This is a linear scale, but probably should be exponential?
            # Current form is y = (-1 / howManySamplesToBias)x + 2
            
#            for i in xrange(howManySamplesToBias-1, -1, -1):
#                s = self.samples[-i]
#                s.weight *= -1.0 / howManySamplesToBias * i + interestingSampleWeightMultiplier
#                s.cumulativeWeight = s.weight + self.samples[-i-1].cumulativeWeight
        
        if len(self.samples) > self.maxSamples * 1.05:
            len_before = len(self.samples)
            self.samples = self.samples[(len(self.samples) - self.maxSamples):]
    
    def drawBatch(self, batchSize):
        if batchSize > len(self.samples):
            raise IndexError('Too few samples (%d) to draw a batch of %d' % (len(self.samples), batchSize))
        return random.sample(self.samples, batchSize)
    
    def drawWeightedBatch(self, batchSize, allowOverSampling=False):
        if not allowOverSampling and batchSize > len(self.samples):
            raise IndexError('Too few samples (%d) to draw a batch of %d' % (len(self.samples), batchSize))

        batch = []
        probe = Sample(None, None, 0, None, 0)
        for i in range(batchSize):
            probe.cumulativeWeight = random.randint(0, self.samples[-1].cumulativeWeight-1)
            realIndex = bisect.bisect_left(self.samples, probe)
            #print('%d %d' % (probe.cumulativeWeight, realIndex))
            batch.append(self.samples[realIndex])
            
        return batch
            

replay = ReplayMemory(1000)

for i in range(1000):
    terminal = 1 if random.randint(0, 50) == 0 else 0
    reward = random.randint(1,5) if random.random() > .9 else 0
    replay.addSample(Sample(None, None, reward, None, terminal))

batch = replay.drawWeightedBatch(1000, True)
for i in range(1000):
  print(replay.samples[i].weight)
   
#for i in range(len(replay.samples)):
#    if replay.samples[i].isInteresting():
#        print('%d  %d' % (replay.samples[i].reward, replay.samples[i].terminal))