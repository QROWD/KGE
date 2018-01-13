import sys,os
import numpy as np

def randn(*args):
  return np.random.randn(*args).astype('f')

def L2_proj(M):
  norm_M = np.linalg.norm(M,axis=1)
  M /=  norm_M[:,None]
  return M

class Triples(object):

  def __init__(self, data):
    self.indexes = np.array(data.keys()).astype(np.int64)
    self.values = np.array(data.values()).astype(np.float32)

class Parameters(object):

  def __init__(self, model, lmbda=0.1, k=100, lr=0.5, epoch=1000, bsize=500, 
    negative=10, valid=500, sgd='adagrad'):

    self.model = model
    self.lmbda = lmbda
    self.k = k
    self.lr = lr
    self.epoch = epoch
    self.batch_size = bsize
    self.neg_ratio = negative
    self.valid_scores = valid
    self.sgd = sgd
