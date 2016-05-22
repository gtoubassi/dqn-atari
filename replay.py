import numpy as np
import bisect
import math
import random

class Sample:
    
    def __init__(self, state1, state1QValues, action, reward, state2, terminal):
        self.state1 = state1
        self.state1QValues = state1QValues
        self.state1QValue = state1QValues[action] if state1QValues is not None else 0
        self.action = action
        self.reward = reward
        self.state2 = state2
        self.state2QValue = 0
        self.terminal = terminal
        self.weight = 1
        self.cumulativeWeight = 1

    def isInteresting(self):
        return self.terminal or self.reward != 0

    def __cmp__(self, obj):
        return self.cumulativeWeight - obj.cumulativeWeight


class ReplayMemory:
    
    def __init__(self, args):
        self.samples = []
        self.maxSamples = args.replay_capacity
        self.prioritizedReplay = args.prioritized_replay
        self.numInterestingSamples = 0
        self.batchesDrawn = 0

    def numSamples():
        return len(self.samples)

    def addSample(self, sample):
        if len(self.samples) > 0 and not self.samples[-1].terminal:
            self.samples[-1].state2QValue = np.max(sample.state1QValues)
        self.samples.append(sample)
        self._updateWeightsForNewlyAddedSample()
        self._truncateListIfNecessary()

    def _updateWeightsForNewlyAddedSample(self):
        if len(self.samples) > 2:
            self.samples[-2].cumulativeWeight -= self.samples[-2].weight
            self.samples[-2].weight = abs(self.samples[-2].state1QValue - self.samples[-2].state2QValue)
            self.samples[-2].cumulativeWeight += self.samples[-2].weight

        if len(self.samples) > 1:
            self.samples[-1].cumulativeWeight = self.samples[-1].weight + self.samples[-2].cumulativeWeight

    def _truncateListIfNecessary(self):
        # premature optimizastion alert :-), don't truncate on each
        # added sample since (I assume) it requires a memcopy of the list (probably 8mb)
        if len(self.samples) > self.maxSamples * 1.05:
            truncatedWeight = 0
            # Before truncating the list, correct self.numInterestingSamples, and prepare
            # for correcting the cumulativeWeights of the remaining samples
            for i in range(self.maxSamples, len(self.samples)):
                truncatedWeight += self.samples[i].weight
                if self.samples[i].isInteresting():
                    self.numInterestingSamples -= 1

            # Truncate the list
            self.samples = self.samples[(len(self.samples) - self.maxSamples):]
            
            # Correct cumulativeWeights
            for sample in self.samples:
                sample.cumulativeWeight -= truncatedWeight
    
    def drawBatch(self, batchSize):
        if batchSize > len(self.samples):
            raise IndexError('Too few samples (%d) to draw a batch of %d' % (len(self.samples), batchSize))
        
        self.batchesDrawn += 1
        
        if self.prioritizedReplay:
            return self._drawPrioritizedBatch(batchSize)
        else:
            return random.sample(self.samples, batchSize)

    # The nature paper doesn't do this but they mention the idea.
    # This particular approach and the weighting I am using is a total
    # uninformed fabrication on my part.  There is probably a more
    # principled way to do this
    def _drawPrioritizedBatch(self, batchSize):
        batch = []
        probe = Sample(None, None, 0, 0, None, False)
        while len(batch) < batchSize:
            probe.cumulativeWeight = random.uniform(0, self.samples[-1].cumulativeWeight)
            index = bisect.bisect_right(self.samples, probe, 0, len(self.samples) - 1)
            sample = self.samples[index]
            #sample.weight = max(1, .8 * sample.weight)
            if sample not in batch:
                batch.append(sample)

        if self.batchesDrawn % 100 == 0:
            cumulative = 0
            for sample in self.samples:
                cumulative += sample.weight
                sample.cumulativeWeight = cumulative
        return batch
    
    def _printBatchWeight(self, batch):
        batchWeight = 0
        for i in range(0, len(batch)):
            batchWeight += batch[i].weight
        print('batch weight: %f' % batchWeight)
