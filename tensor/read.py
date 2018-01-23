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

  aux = range(folds)
  tmp = range(1, folds) + range(1)

  test = [dict(data[idx == i,:]) for i in aux]
  valid = [dict(data[idx == i,:]) for  i in tmp]
  train = [dict(data[(idx != aux[i]) & (idx != tmp[i]),:]) for i in range(folds)]

  test = [Triples(i) for i in test]
  valid = [Triples(i) for i in valid]
  train = [Triples(i) for i in train]
  return train, valid, test
