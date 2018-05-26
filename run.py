import argparse
import os, pickle
import tensor

from tensor.experiment import *
from tensor.read import *

if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Knowledge Graph Embedding for Link Prediction'
  )

  parser.add_argument('type', metavar='', 
    help='Evaluate or predict the model: {evaluation, prediction}')
  parser.add_argument('--model', metavar='', 
    help='model to run: {Complex, CP, RESCAL, DistMult, TransE}')
  parser.add_argument('--lmbda', type=float, default=0.01, metavar='', 
    help='value of lambda  (default: 0.1)')
  parser.add_argument('--data', metavar='', help='dataset to be used')
  parser.add_argument('--k', type=int, default=150, metavar='', 
    help='embedding size (default: 50)')
  parser.add_argument('--lr', type=float, default=0.5, metavar='', 
    help='Learning rate (default: 0.5)')
  parser.add_argument('--epoch', type=int, default=1000, metavar='', 
    help='Number of epochs (default: 1000)')
  parser.add_argument('--bsize', type=int, default=1000, metavar='', 
    help='Number of examples in the batch sample (default: 500)')
  parser.add_argument('--nsize', type=int, default=1, metavar='', 
    help='Number of negative examples generated (default: 10)')
  parser.add_argument('--folds', type=int, default=10, metavar='', 
    help='Number of k-fold cross validation (default: 10)')
  parser.add_argument('--rand', default=1234, type=int, metavar='',
    help='Set the random seed (default: 1234')

  args = parser.parse_args()

  _, ext = os.path.splitext(args.data)
  np.random.seed(args.rand)

  table, entities, relations = read(args.data, ext)

  print("Nb entities: " + str(len(entities)))
  print("Nb relations: " + str(len(relations)))
  print("Nb triples: " + str(len(table)))

  if(args.type == 'evaluation'):

    print("Technique: " + str(args.model))
    print("Embedding size: " + str(args.k))
    print("Learning rate: " + str(args.lr))
    print("Number of Epochs: " + str(args.epoch))
    print("Negatives ratio: " + str(args.nsize))
    print("Batch size: " + str(args.bsize))

    data = byIndex(table, entities, relations)
    train, test = kfold(data, args.folds)

    param = Parameters(model=args.model, lmbda=args.lmbda, k=args.k, lr=args.lr,
      epoch=args.epoch, bsize=args.bsize, nsize=args.nsize)

    m = []
    for i in range(args.folds):
      print("Fold " + str(i+1) + ":")
      exp = Experiment(train[i], test[i], entities, relations, param)
      exp.evaluation()
      m.append(exp)

    with open("model.txt", "wb") as fp:
      pickle.dump(best(m), fp)

  else:

    with open('model.txt', 'rb') as fp:
      m = pickle.load(fp)

    print("Technique: " + str(m.param.model))
    print("Embedding size: " + str(m.param.k))
    print("Learning rate: " + str(m.param.lr))
    print("Number of Epochs: " + str(m.param.epoch))
    print("Negatives ratio: " + str(m.param.nsize))
    print("Batch size: " + str(m.param.bsize))

    test = Triples(byIndex(table, m.entities, m.relations))
    res = m.prediction(test)
    np.savetxt("out.csv", res)
