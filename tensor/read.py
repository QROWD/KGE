import numpy as np

from rdflib.graph import Graph
from tools import *

def utf8(lst):
  return [unicode(elem).encode('utf-8') for elem in lst]

def rdf(file):

  g = Graph()
  g.parse(file, format="nt")

  data = []
  for s, p, o in g:
    data.append(utf8(["" + s,"" + p,"" + o]))

  data = np.array(data)
  entities = (np.unique((data[:,0], data[:,2]))).tolist()
  relations = (np.unique(data[:,1])).tolist()
  return data, entities, relations

def csv(file):

  data = np.genfromtxt(file, delimiter="\t", dtype='|S146')
  entities = (np.unique((data[:,0], data[:,2]))).tolist()
  relations = (np.unique(data[:,1])).tolist()
  return data, entities, relations

def read(file, ext):

  if(ext == ".nt"):
    return rdf(file)
  return csv(file)

def byIndex(table, entities, relations):

  data = dict()
  for s, p, o in table:
    s, p, o = entities.index(s), relations.index(p), entities.index(o)
    data[(s, p, o)] = 1

  return np.array(data.items())

def kfold(data, folds=10):

  idx = np.repeat(range(folds), int(len(data)/folds))
  data = data[0:len(idx),:]

  train = [Triples(data[(idx != i),:]) for i in range(folds)]
  test = [Triples(data[idx == i,:]) for i in range(folds)]
  return train, test
