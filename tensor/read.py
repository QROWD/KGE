import numpy as np

from experiment import *

def read(filename, split="\t"):

  table = np.genfromtxt(filename, delimiter=split, dtype='|S146')
  entities = (np.unique((table[:,0], table[:,2]))).tolist()
  relations = (np.unique(table[:,1])).tolist()

  data = dict()
  for s, p, o in table:
    s, p, o = entities.index(s), relations.index(p), entities.index(o)
    data[(s, p, o)] = 1

  return np.array(data.items()), len(entities), len(relations)

def kcv(data, k=10):

  index = np.repeat(range(k), int(len(data)/k))
  data = data[0:len(index),:]

  test = [dict(data[index == i,:]) for i in range(k)]
  valid = [dict(data[index == i,:]) for  i in range(1, k) + range(1)]
  train = [dict(data[(index != i) & (index != i-1),:]) for i in range(1, k)]
  return train, valid, test

def load(path, file):

  data, entities, relations = read(path + '/datasets/' + file)
  train, valid, test = kcv(data, 10)

  test = [Triples(i) for i in test]
  valid = [Triples(i) for i in valid]
  train = [Triples(i) for i in train]

  return data, train, valid, test, entities, relations
