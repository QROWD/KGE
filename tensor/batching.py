import numpy as np

class Batch(object):

  def __init__(self, data, entities, bsize=100, nsize=10):
    self.data = data
    self.bsize = bsize
    self.entities = entities
    self.nsize = int(nsize)

    idx = np.random.randint(0, len(self.data.values), self.bsize)
    self.data.indexes = self.data.indexes[idx,:]
    self.data.values = self.data.values[idx]

  def __call__(self):

    indexes = []
    values = []

    for i in range(self.bsize):
      for j in range(self.nsize):
        aux = self.data.indexes[i]
        if np.random.random_sample() < 0.5:
          aux[0] = np.random.randint(0, self.entities, 1)[0]
        else:
          aux[2] = np.random.randint(0, self.entities, 1)[0]
        indexes.append(aux)
        values.append(-1)

    indexes = np.array(indexes).astype(np.int64)
    values = np.array(values).astype(np.float32)

    indexes = np.concatenate((indexes, self.data.indexes))
    values = np.concatenate((values, self.data.values))

    train = [values, indexes[:,0], indexes[:,1], 
      indexes[:,2]]

    return train
