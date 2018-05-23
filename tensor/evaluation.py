import operator

from tools import *

class Scorer(object):

  def __init__(self, train, test):

    self.obj = {}
    self.sub = {}

    self.update(train.indexes)
    self.update(test.indexes)

  def update(self, data):

    for i, j, k in data:

      if (i, j) not in self.obj:
        self.obj[(i, j)] = [k]
      elif k not in self.obj[(i, j)]:
        self.obj[(i, j)].append(k)

      if (j, k) not in self.sub:
        self.sub[(j, k)] = [i]
      elif i not in self.sub[(j, k)]:
        self.sub[(j, k)].append(i)

  def evaluation(self, model, test):
    pred = model.predict(test.indexes)
    return np.concatenate((self.head(model, test), self.tail(model, test)))

  def head(self, model, test):

    rank = np.empty(len(test.values))
    for a, (i, j, k) in enumerate(test.indexes):

      sub = model.subjects(j, k)
      aux = 1 + np.sum(sub > sub[i])
      rank[a] = aux - np.sum(sub[self.sub[(j,k)]] > sub[i])

    return rank

  def tail(self, model, test):

    rank = np.empty(len(test.values))
    for a, (i, j, k) in enumerate(test.indexes):

      obj = model.objects(i, j)
      aux = 1 + np.sum(obj > obj[k])
      rank[a] = aux - np.sum(obj[self.obj[(i, j)]] > obj[k])

    return rank
