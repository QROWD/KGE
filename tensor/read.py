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

  data = np.array(data.items())
  return data, len(entities), len(relations)

def load(path, file):

  data, entities, relations = read(path + '/datasets/' + file)
  return data, entities, relations

def kcv(data, folds=10):

  idx = np.repeat(range(folds), int(len(data)/folds))
  data = data[0:len(idx),:]

  train = [dict(data[(idx != i),:]) for i in range(folds)]
  test = [dict(data[idx == i,:]) for i in range(folds)]

  train = [Triples(i) for i in train]
  test = [Triples(i) for i in test]
  return train, test
