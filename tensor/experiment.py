import numpy as np

import models
from evaluation import *
from tools import *

class Experiment(object):

  def __init__(self, train, valid, test, entities, relations, param):

    self.train = train
    self.valid = valid
    self.test = test

    self.n_entities = entities
    self.n_relations = relations

    self.scorer = Scorer(train, valid, test)
    self.model = vars(models)[param.model]()
    self.param = param

    self.results = Results()

  def induce(self):
    print("Inducing")

    self.model.fit(self.train, self.valid, self.param, self.n_entities, 
      self.n_relations, self.scorer)

  def evaluate(self):
    print("Evaluating")
    res = self.scorer.compute(self.model, self.test)
    self.results.add(res)
    self.results.measures()
