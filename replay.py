import bisect
import math
import random

class Sample:
    
    def __init__(self, state1, action, reward, state2, terminal):
        self.state1 = state1
        self.action = action
        self.reward = reward
        self.state2 = state2
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
        self.samples.append(sample)
        self._updateWeightsForNewlyAddedSample()
        self._truncateListIfNecessary()

    def _updateWeightsForNewlyAddedSample(self):
        if len(self.samples) > 1:
            self.samples[-1].cumulativeWeight = self.samples[-1].weight + self.samples[-2].cumulativeWeight

        if self.samples[-1].isInteresting():
            self.numInterestingSamples += 1
            
            # Boost the neighboring samples.  How many samples?  Roughly the number of samples
            # that are "uninteresting".  Meaning if interesting samples occur 3% of the time, then boost 33
            uninterestingSampleRange = max(1, len(self.samples) / max(1, self.numInterestingSamples))
            for i in range(uninterestingSampleRange, 0, -1):
                index = len(self.samples) - i
                if index < 1:
                    break
                # This is an exponential that ranges from 3.0 to 1.01 over the domain of [0, uninterestingSampleRange]
                # So the interesting sample gets a 3x boost, and the one furthest away gets a 1% boost
                boost = 1.0 + 3.0/(math.exp(i/(uninterestingSampleRange/6.0)))
                self.samples[index].weight *= boost
                self.samples[index].cumulativeWeight = self.samples[index].weight + self.samples[index - 1].cumulativeWeight
    
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
        probe = Sample(None, 0, 0, None, False)
        while len(batch) < batchSize:
            probe.cumulativeWeight = random.uniform(0, self.samples[-1].cumulativeWeight)
            index = bisect.bisect_right(self.samples, probe, 0, len(self.samples) - 1)
            sample = self.samples[index]
            sample.weight = max(1, .8 * sample.weight)
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
