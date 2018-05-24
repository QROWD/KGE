import numpy as np

class Batch(object):

  def __init__(self, data, entities, bsize, nsize):
    self.data = data
    self.bsize = bsize
    self.entities = entities
    self.nsize = nsize

    self.indexes = self.data.indexes
    self.values = self.data.values

  def __call__(self):

    idx = np.random.randint(0, len(self.data.values), self.bsize)
    self.indexes = self.data.indexes[idx,:]
    self.values = self.data.values[idx]

    idx = np.repeat(range(len(self.indexes)), self.nsize)
    values = np.repeat(-1, len(self.indexes) * self.nsize)
    indexes = self.indexes[idx,:]

    for i in range(len(indexes)):
      if np.random.random_sample() < 0.5:
        indexes[i,0] = np.random.randint(0, self.entities, 1)[0]
      else:
        indexes[i,2] = np.random.randint(0, self.entities, 1)[0]

    self.indexes = np.concatenate((self.indexes, indexes), axis=0)
    self.values = np.concatenate((self.values, values), axis=0)

    self.indexes = self.indexes.astype(np.int64)
    self.values = self.values.astype(np.float32)

    return [self.values, self.indexes[:,0], self.indexes[:,1], self.indexes[:,2]]
