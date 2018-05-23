import numpy as np
from random import randint

class Batch(object):

  def __init__(self, data, entities, bsize=100, nsize=10):
    data = data
    bsize = bsize
    entities = entities
    nsize = int(nsize)

    idx = np.random.randint(0, len(data.values), bsize)
    data.indexes = data.indexes[idx,:]
    data.values = data.values[idx]

  def __call__(self):

    indexes = []
    values = []

    for i in range(bsize):
      for j in range(nsize):
        aux = (data.indexes[i]).tolist()
        if np.random.random_sample() < 0.5:
          aux[0] = randint(0, entities)
        else:
          aux[2] = randint(0, entities)
        indexes.append(aux)
        values.append(-1)

    indexes = np.array(indexes).astype(np.int64)
    values = np.array(values).astype(np.float32)

    indexes = np.concatenate((indexes, data.indexes))
    values = np.concatenate((values, data.values))

    return [values, indexes[:,0], indexes[:,1], indexes[:,2]]
