import numpy as np
import random as rd

from experiment import *

def read(filename, split="\t"):

  table = np.genfromtxt(filename, delimiter=split, dtype='|S146')
  entities = (np.unique((table[:,0], table[:,2]))).tolist()
  relations = (np.unique(table[:,1])).tolist()
  data = dict()

  for s, p, o in table:
    s = entities.index(s)
    p = relations.index(p)
    o = entities.index(o)
    data[(s, p, o)] = 1

  return data, len(entities), len(relations)

def cv(data, rate):
   train = dict(rd.sample(data.items(), int(len(data)*(1 -rate))))
   valid = dict(rd.sample(data.items(), int(len(data)*rate)))
   test = dict(rd.sample(data.items(), int(len(data)*rate)))
   return train, valid, test

def load(path, file):

  data, entities, relations = read(path + '/datasets/' + file)
  train, valid, test = cv(data, 0.1)

  train = Triples(train)
  valid = Triples(valid)
  test = Triples(test)

  return data, train, valid, test, entities, relations
