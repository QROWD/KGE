import operator

from tools import *

class Result(object):

  def __init__(self, ranks, raw_ranks):
    self.ranks = ranks
    self.mrr = np.mean(1.0 / ranks)
    self.raw_mrr = np.mean(1.0 / raw_ranks)

class Results(object):

  def __init__(self):
    self.res = list()

  def measures(self, res):

    self.res.append(res)
    mrr = np.mean([res.mrr for res in self.res])
    raw_mrr = np.mean([res.raw_mrr for res in self.res])
    rank = [res.ranks for res in self.res]

    h1 = np.mean([(np.sum(r <= 1)) / float(len(r)) for r in rank])
    h3 = np.mean([(np.sum(r <= 3)) / float(len(r)) for r in rank])
    h10= np.mean([(np.sum(r <= 10))/ float(len(r)) for r in rank])

    print("MRR\tRMRR\tH@1\tH@3\tH@10")
    print("%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (mrr, raw_mrr, h1, h3, h10))

    return (mrr, raw_mrr, h1, h3, h10)

class Scorer(object):

  def __init__(self, train, test):

    data = np.concatenate((train.indexes, test.indexes), axis=0)
    self.obj = self.sub = {}

    for i, j, k in data:
      self.obj[(i, j)] = k
      self.sub[(j, k)] = i

  def compute(self, model, test):

    nb_test = len(test.values)
    nrank = np.empty(2*nb_test)
    rrank = np.empty(2*nb_test)

    for a, (i, j, k) in enumerate(test.indexes):

      res_obj = model.eval_o(i, j)
      rrank[a] = 1 + np.sum(res_obj > res_obj[k])
      nrank[a] = rrank[a] - np.sum(res_obj[self.obj[(i,j)]] > res_obj[k])

      res_sub = model.eval_s(j, k)
      rrank[nb_test + a] = 1 + np.sum(res_sub > res_sub[i])
      nrank[nb_test + a] = rrank[nb_test + a] - np.sum(
        res_sub[self.sub[(j,k)]] > res_sub[i])

    return Result(nrank, rrank)
