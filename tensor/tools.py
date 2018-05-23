import sys, os
import numpy as np

def randn(*args):
  return np.random.randn(*args).astype('f')

def L2(M):
  norm_M = np.linalg.norm(M, axis=1)
  M /=  norm_M[:, None]
  return M

def best(model):

  acc = idx = 0
  for i in range(len(model)):
    if(acc < model[i].result[0]):
      acc = model[i].result[0]
      idx = i

  return model[idx]

def measures(rank):

  mrr = np.mean(1.0 / rank)

  h1  = (np.sum(rank <= 1))  / float(len(rank))
  h3  = (np.sum(rank <= 3))  / float(len(rank))
  h10 = (np.sum(rank <= 10)) / float(len(rank))

  print("MRR\tH@1\tH@3\tH@10")
  print("%0.3f\t%0.3f\t%0.3f\t%0.3f" % \
    (mrr, h1, h3, h10))

  return (mrr, h1, h3, h10)

class Triples(object):

  def __init__(self, data):
    data = dict(data)
    self.indexes = np.array(data.keys()).astype(np.int64)
    self.values = np.array(data.values()).astype(np.float32)

class Parameters(object):

  def __init__(self, model='Complex', lmbda=0.1, k=100, lr=0.5, epoch=1000, 
    bsize=500, nsize=10, sgd='adagrad'):

    self.model = model
    self.lmbda = lmbda
    self.k = k
    self.lr = lr
    self.epoch = epoch
    self.bsize = bsize
    self.nsize = nsize
    self.sgd = sgd

