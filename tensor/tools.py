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

  def __init__(self, model, lmbda=0.1, embedding_size = 100, learning_rate=0.5, 
    max_iter=1000, batch_size=500, neg_ratio=10, valid_scores=500, 
    learning_policy='adagrad'):

    self.model = model
    self.lmbda = lmbda
    self.embedding_size = embedding_size
    self.batch_size = batch_size
    self.max_iter = max_iter
    self.learning_rate = learning_rate
    self.neg_ratio = neg_ratio
    self.valid_scores = valid_scores
    self.learning_policy = learning_policy
