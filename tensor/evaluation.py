import operator

from tracelog import *
from tools import *

class Result(object):

  def __init__(self, ranks, raw_ranks):
    self.ranks = ranks
    self.mrr = np.mean(1.0 / ranks)
    self.raw_mrr = np.mean(1.0 / raw_ranks)

class Results(object):

  def __init__(self):
    self.res = list()

  def add(self, res):
    self.res.append(res)

  def measures(self):

    mrr = np.mean([res.mrr for res in self.res])
    raw_mrr = np.mean([res.raw_mrr for res in self.res])
    ranks_list = [res.ranks for res in self.res]

    hits1 = np.mean([(np.sum(ranks <= 1)) / float(len(ranks)) for ranks in ranks_list])
    hits3 = np.mean([(np.sum(ranks <= 3)) / float(len(ranks)) for ranks in ranks_list])
    hits10= np.mean([(np.sum(ranks <= 10))/ float(len(ranks)) for ranks in ranks_list])

    logger.info("MRR\tRMRR\tH@1\tH@3\tH@10")
    logger.info("%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (mrr, raw_mrr, hits1, hits3, hits10))

    return (mrr, raw_mrr, hits1, hits3, hits10)

class Scorer(object):

  def __init__(self, train, valid, test):

    self.obj = {}
    self.sub = {}

    self.update(train.indexes)
    self.update(test.indexes)
    self.update(valid.indexes)

  def update(self, triples):
    for i,j,k in triples:
      if (i,j) not in self.obj:
        self.obj[(i,j)] = [k]
      elif k not in self.obj[(i,j)]:
        self.obj[(i,j)].append(k)

      if (j,k) not in self.sub:
        self.sub[(j,k)] = [i]
      elif i not in self.sub[(j,k)]:
        self.sub[(j,k)].append(i)

  def compute(self, model, data):

    nb_test = len(data.values)
    ranks = np.empty(2 * nb_test)
    raw_ranks = np.empty(2 * nb_test)

    for a,(i,j,k) in enumerate(data.indexes[:nb_test,:]):

      res_obj = model.eval_o(i,j)
      raw_ranks[a] = 1 + np.sum( res_obj > res_obj[k] )
      ranks[a] = raw_ranks[a] - np.sum(res_obj[self.obj[(i,j)]] > res_obj[k])

      res_sub = model.eval_s(j,k)
      raw_ranks[nb_test + a] = 1 + np.sum( res_sub > res_sub[i] )
      ranks[nb_test + a] = raw_ranks[nb_test + a] - np.sum(
        res_sub[self.sub[(j,k)]] > res_sub[i] )

    return Result(ranks, raw_ranks)
