import scipy
import scipy.io
import random as rd

from experiment import *

def parse(filename, line):
  sub, rel, obj = line.strip().split("\t")
  return sub, obj, rel, 1

def read(filename):

  entities_indexes= dict()
  entities = set()
  next_ent = 0

  relations_indexes= dict()
  relations= set()
  next_rel = 0

  data = dict()

  with open(filename) as f:
    lines = f.readlines()

  for i,line in enumerate(lines):

    sub, obj, rel, val = parse(filename, line)

    if sub in entities:
      sub_ind = entities_indexes[sub]
    else:
      sub_ind = next_ent
      next_ent += 1
      entities_indexes[sub] = sub_ind
      entities.add(sub)

    if obj in entities:
      obj_ind = entities_indexes[obj]
    else:
      obj_ind = next_ent
      next_ent += 1
      entities_indexes[obj] = obj_ind
      entities.add(obj)

    if rel in relations:
      rel_ind = relations_indexes[rel]
    else:
      rel_ind = next_rel
      next_rel += 1
      relations_indexes[rel] = rel_ind
      relations.add(rel)

    data[(sub_ind, rel_ind, obj_ind)] = val

  return data, entities_indexes, relations_indexes

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
