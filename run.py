import sys, os
import argparse
import pickle
import tensor

import tensor.tools as tools
from tensor.read import *

if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Knowledge Graph Embedding for Link Prediction'
  )

  parser.add_argument('type', metavar='', 
    help='Evaluate or predict the model: {evaluation, prediction}')
  parser.add_argument('--model', metavar='', 
    help='model to run: {Complex, CP, Rescal, DistMult, TransE}')
  parser.add_argument('--lmbda', type=float, default=0.1, metavar='', 
    help='value of lambda  (default: 0.1)')
  parser.add_argument('--data', metavar='', help='dataset to be used')
  parser.add_argument('--k', type=int, default=50, metavar='', 
    help='embedding size (default: 50)')
  parser.add_argument('--lr', type=float, default=0.5, metavar='', 
    help='Learning rate (default: 0.5)')
  parser.add_argument('--epoch', type=int, default=1000, metavar='', 
    help='Number of epochs (default: 1000)')
  parser.add_argument('--bsize', type=int, default=500, metavar='', 
    help='Number of examples in the batch sample (default: 500)')
  parser.add_argument('--negative', type=int, default=10, metavar='', 
    help='Number of negative examples generated (default: 10)')
  parser.add_argument('--folds', type=int, default=10, metavar='', 
    help='Number of k-fold cross validation (default: 10)')
  parser.add_argument('--rand', default=1234, type=int, metavar='',
    help='Set the random seed (default: 1234')

  path = os.path.dirname(os.path.realpath(os.path.basename(__file__)))

  args = parser.parse_args()
  np.random.seed(args.rand)

  if(args.type == 'evaluation'):

    data, entities, relations = original(path, args.data)
    train, test = kcv(data, args.folds)

    print("Nb entities: " + str(len(entities)))
    print("Nb relations: " + str(len(relations)))
    print("Nb triples: " + str(len(data)))
    print("Technique: " + str(args.model))
    print("Learning rate: " + str(args.lr))
    print("Max epochs: " + str(args.epoch))
    print("Generated negatives ratio: " + str(args.negative))
    print("Batch size: " + str(args.bsize))

    param = Parameters(model=args.model, lmbda=args.lmbda, k=args.k, lr=args.lr,
      epoch=args.epoch, bsize=args.bsize, negative=args.negative)

    model = []
    for i in range(args.folds):
      print("Fold " + str(i+1) + ":")
      exp = Experiment(train[i], test[i], entities, relations, param)
      exp.induce()
      exp.evaluate()
      model.append(exp)

    acc = idx = 0
    for i in range(len(model)):
      if(acc < model[i].results.res[0].mrr):
        acc = model[i].results.res[0].mrr
        idx = i

    with open("model.txt", "wb") as fp:
      pickle.dump(model[idx], fp)

  else: 
    print(0)