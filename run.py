import sys, os

import tensor
import tensor.tracelog as log
import tensor.tools as tools
from tensor.read import *

if __name__ == "__main__":

  path = os.path.dirname(os.path.realpath( os.path.basename(__file__)))

  data, train, valid, test, entities, relations = load(path, 'wn18.txt')

  #param = Parameters("Complex_Logistic", embedding_size=100, max_iter=1000, valid_scores=100)
  #param = Parameters("DistMult_Logistic", embedding_size=100, max_iter=1000, valid_scores=100)
  #param = Parameters("CP_Logistic", embedding_size=100, max_iter=1000, valid_scores=100)
  #param = Parameters("TransE_L1", embedding_size=100, max_iter=1000, valid_scores=100)
  #param = Parameters("TransE_L2", embedding_size=100, max_iter=1000, valid_scores=100)

  param = Parameters("Complex_Logistic", embedding_size=100, max_iter=1000, valid_scores=500)
  model = Experiment(data, train, valid, test, entities, relations, param)

  log.logger.info( "Technique: " + str(param.model))
  log.logger.info( "Learning rate: " + str(param.learning_rate))
  log.logger.info( "Max iter: " + str(param.max_iter))
  log.logger.info( "Generated negatives ratio: " + str(param.neg_ratio))
  log.logger.info( "Batch size: " + str(param.batch_size))

  model.induce()
  model.evaluate()

  param = Parameters("DistMult_Logistic", embedding_size=100, max_iter=1000, valid_scores=500)
  model = Experiment(data, train, valid, test, entities, relations, param)

  log.logger.info( "Technique: " + str(param.model))
  log.logger.info( "Learning rate: " + str(param.learning_rate))
  log.logger.info( "Max iter: " + str(param.max_iter))
  log.logger.info( "Generated negatives ratio: " + str(param.neg_ratio))
  log.logger.info( "Batch size: " + str(param.batch_size))

  model.induce()
  model.evaluate()

  param = Parameters("Polyadic_Logistic", embedding_size=100, max_iter=1000, valid_scores=500)
  model = Experiment(data, train, valid, test, entities, relations, param)

  log.logger.info( "Technique: " + str(param.model))
  log.logger.info( "Learning rate: " + str(param.learning_rate))
  log.logger.info( "Max iter: " + str(param.max_iter))
  log.logger.info( "Generated negatives ratio: " + str(param.neg_ratio))
  log.logger.info( "Batch size: " + str(param.batch_size))

  model.induce()
  model.evaluate()

  param = Parameters("TransE_L1", embedding_size=100, max_iter=1000, valid_scores=500)
  model = Experiment(data, train, valid, test, entities, relations, param)

  log.logger.info( "Technique: " + str(param.model))
  log.logger.info( "Learning rate: " + str(param.learning_rate))
  log.logger.info( "Max iter: " + str(param.max_iter))
  log.logger.info( "Generated negatives ratio: " + str(param.neg_ratio))
  log.logger.info( "Batch size: " + str(param.batch_size))

  model.induce()
  model.evaluate()

  param = Parameters("TransE_L2", embedding_size=100, max_iter=1000, valid_scores=500)
  model = Experiment(data, train, valid, test, entities, relations, param)

  log.logger.info( "Technique: " + str(param.model))
  log.logger.info( "Learning rate: " + str(param.learning_rate))
  log.logger.info( "Max iter: " + str(param.max_iter))
  log.logger.info( "Generated negatives ratio: " + str(param.neg_ratio))
  log.logger.info( "Batch size: " + str(param.batch_size))

  model.induce()
  model.evaluate()
