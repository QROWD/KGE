import numpy as np

from experiment import *

def read(filename, split="\t"):

  data = np.genfromtxt(filename, delimiter=split, dtype='|S146')
  entities = (np.unique((data[:,0], data[:,2]))).tolist()
  relations = (np.unique(data[:,1])).tolist()
  return data, entities, relations

def byIndex(table, entities, relations):

  data = dict()
  for s, p, o in table:
    s, p, o = entities.index(s), relations.index(p), entities.index(o)
    data[(s, p, o)] = 1

  return np.array(data.items())

def splitted(path, file):

  aux = read(path + 'splitted/' + file + '.train.txt')
  tmp = read(path + 'splitted/' + file + '.test.txt')

  entities = (np.unique(aux[1] + tmp[1])).tolist()
  relations = (np.unique(aux[2] + tmp[2])).tolist()

  train = byIndex(aux[0], entities, relations)
  test  = byIndex(tmp[0], entities, relations)
  return [train], [test], len(entities), len(relations)

def original(path, file):

  data, entities, relations = read(path + 'original/' + file + '.txt')
  data = byIndex(data, entities, relations)
  return data, len(entities), len(relations)

def kcv(data, folds=10):

  idx = np.repeat(range(folds), int(len(data)/folds))
  data = data[0:len(idx),:]

  train = [dict(data[(idx != i),:]) for i in range(folds)]
  test = [dict(data[idx == i,:]) for i in range(folds)]

  train = [Triples(i) for i in train]
  test = [Triples(i) for i in test]
  return train, test
