import numpy as np

import models
from evaluation import *
from tools import *

class Experiment(object):

  def __init__(self, train, test, entities, relations, param):

    self.train = train
    self.test = test

    self.entities = entities
    self.relations = relations

    self.scorer = Scorer(train, test)
    self.model = vars(models)[param.model]()
    self.results = Results()
    self.param = param

  def induce(self):

    print("Inducing")
    self.model.fit(self.train, self.entities, self.relations, self.param)

  def evaluate(self):

    print("Evaluating")
    res = self.scorer.compute(self.model, self.test)
    self.results.measures(res)
