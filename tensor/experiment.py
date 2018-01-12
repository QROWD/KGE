import uuid
import time
import subprocess
import xml.dom.minidom
import cPickle as pickle
import numpy as np

import models
from evaluation import *
from log import *
from tools import *

class Experiment(object):

  def __init__(self, data, train, valid, test, entities, relations, param):

    self.train = train
    self.valid = valid
    self.test = test

    self.train_tensor = None
    self.train_mask = None
    self.positives_only = True
    self.n_entities = len(entities)
    self.n_relations = len(relations)

    logger.info("Nb entities: " + str(self.n_entities))
    logger.info("Nb relations: " + str(self.n_relations))
    logger.info("Nb triples: " + str(len(data)))
    
    self.scorer = Scorer(train, valid, test)
    self.model = vars(models)[param.model]()
    self.param = param

    self.valid_results = Results()
    self.results = Results()

  def induce(self):
    logger.info("Inducing")

    self.model.fit(self.train, self.valid, self.param, self.n_entities, 
      self.n_relations, self.n_entities, self.scorer)

    res = self.scorer.compute(self.model, self.valid)
    self.valid_results.add(res)

  def evaluate(self):
    logger.info("Evaluating")
    res = self.scorer.compute(self.model, self.test)
    self.results.add(res)
    self.results.measures()
