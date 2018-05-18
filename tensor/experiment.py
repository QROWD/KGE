import numpy as np

import models
from evaluation import *
from tools import *

class Experiment(object):

  def __init__(self, train, test, entities, relations, param):

    self.train = train
    self.test = test

    self.orig_entities = entities
    self.orig_relations = relations

    self.entities = len(entities)
    self.relations = len(relations)

    self.scorer = Scorer(train, test)
    self.model = vars(models)[param.model]()
    self.results = Results()
    self.param = param

  def induce(self):
    print("Induction")
    self.model.fit(self.train, self.entities, self.relations, self.param)

  def evaluate(self):
    print("Evaluation")
    res = self.scorer.evaluate(self.model, self.test)
    self.results.measures(res)

  def prediction(self, test):
    print("Prediction")
    res = self.scorer.predict(self.model, test)
    return(res)
