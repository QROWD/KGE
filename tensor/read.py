import numpy as np

from experiment import *

def read(filename, split="\t"):

  data = np.genfromtxt(filename, delimiter=split, dtype='string', comments=None)
  entities = (np.unique((data[:,0], data[:,2]))).tolist()
  relations = (np.unique(data[:,1])).tolist()
  return data, entities, relations

def byIndex(table, entities, relations):

  data = dict()
  for s, p, o in table:
    s, p, o = entities.index(s), relations.index(p), entities.index(o)
    data[(s, p, o)] = 1

  return np.array(data.items())

def original(path, file):

  table, entities, relations = read(path + '/datasets/' + file)
  data = byIndex(table, entities, relations)
  return data, entities, relations

def kcv(data, folds=10):

  idx = np.repeat(range(folds), int(len(data)/folds))
  data = data[0:len(idx),:]

  train = [Triples(data[(idx != i),:]) for i in range(folds)]
  test = [Triples(data[idx == i,:]) for i in range(folds)]
  return train, test
