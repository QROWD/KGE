import numpy as np

import models
from evaluation import *
from tracelog import *
from tools import *

class Experiment(object):

  def __init__(self, data, train, valid, test, entities, relations, param):

    self.train = train
    self.valid = valid
    self.test = test

    self.n_entities = len(entities)
    self.n_relations = len(relations)

    logger.info("Nb entities: " + str(self.n_entities))
    logger.info("Nb relations: " + str(self.n_relations))
    logger.info("Nb triples: " + str(len(data)))
    
    self.scorer = Scorer(train, valid, test)
    self.model = vars(models)[param.model]()
    self.param = param

    self.results = Results()

  def induce(self):
    logger.info("Inducing")

    self.model.fit(self.train, self.valid, self.param, self.n_entities, 
      self.n_relations, self.scorer)

  def evaluate(self):
    logger.info("Evaluating")
    res = self.scorer.compute(self.model, self.test)
    self.results.add(res)
    self.results.measures()
