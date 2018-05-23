import theano
import theano.tensor as T
import downhill

from batching import *
from tools import *

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_RUN'

class Model(object):

  def __init__(self):
    self.name = self.__class__.__name__

    self.pred = None
    self.pred_compiled = None

    self.regul = None
    self.loss_opt = None
    
    self.ys = T.vector('ys')
    self.rows = T.lvector('rows')
    self.cols = T.lvector('cols')
    self.tubes = T.lvector('tubes') 

    self.n, self.m, self.l, self.k = 0, 0, 0, 0

  def set_dims(self, train, param):
    self.n = max(train.indexes[:,0]) + 1
    self.m = max(train.indexes[:,1]) + 1
    self.l = max(train.indexes[:,2]) + 1
    self.k = param.k

  def pred_symb_vars(self):
    return [self.rows, self.cols, self.tubes]

  def pred_args(self, test):
    return [test[:,0], test[:,1], test[:,2]]

  def batch(self, train, param):

    train = Batch(train, entities=self.n, bsize=param.bsize,
      neg_ratio=param.neg_ratio)
    inputs = [self.ys, self.rows, self.cols, self.tubes]
    return train, inputs

  def start(self):
    params = self.tensors()
    for name, val in params.items():
      setattr(self, name, theano.shared(val, name=name))

  def restart(self):
    params = self.tensors()
    for name, val in params.items():
      getattr(self, name).set_value(val, borrow=True)

  def setup(self, train, param):

    if self.loss_opt is None:
      self.start()
      self.lossfun()
      self.pred_compiled = theano.function(self.pred_symb_vars(), self.pred)
    else:
      self.restart()

    self.loss_opt = self.loss + param.lmbda * self.regul

  def fit(self, train, entities, relations, param):

    self.n, self.m, self.l, self.k = entities, relations, entities, param.k
    self.setup(train, param)

    vals, symbs = self.batch(train, param)
    opt = downhill.build(param.sgd, loss=self.loss_opt, inputs=symbs, 
      monitor_gradients=True)

    vals = downhill.Dataset(vals, name='train')

    it = 0
    for _ in opt.iterate(vals, None, max_updates=param.epoch,
      validate_every=10, patience=5, max_gradient_norm=1, 
      learning_rate=param.lr):

      it += 1
      if it >= param.epoch:
        break

  def predict(self, test):
    return self.pred_compiled(*self.pred_args(test))

  def tensors(self):
    pass

  def lossfun(self):
    pass

class CP(Model):

  def __init__(self):
    super(CP, self).__init__()
    self.name = self.__class__.__name__

    self.u = None
    self.v = None
    self.w = None

  def tensors(self):

    params = {'u': randn(max(self.n,self.l),self.k),
              'v': randn(self.m,self.k),
              'w': randn(max(self.n,self.l),self.k)}
    return params

  def lossfun(self):

    self.pred = T.sum(self.u[self.rows,:] * self.v[self.cols,:] * 
      self.w[self.tubes,:], 1)

    self.loss = T.sqr(self.ys - self.pred).mean()

    self.regul = T.sqr(self.u[self.rows,:]).mean() \
      + T.sqr(self.v[self.cols,:]).mean() \
      + T.sqr(self.w[self.tubes,:]).mean()

  def objects(self, i, j):
    u = self.u.get_value(borrow=True)
    v = self.v.get_value(borrow=True)
    w = self.w.get_value(borrow=True)
    return (u[i,:] * v[j,:]).dot(w.T)

  def subjects(self, j, k):
    u = self.u.get_value(borrow=True)
    v = self.v.get_value(borrow=True)
    w = self.w.get_value(borrow=True)
    return u.dot(v[j,:] * w[k,:])

class Complex(Model):

  def __init__(self):
    super(Complex, self).__init__()
    self.name = self.__class__.__name__

    self.e1 = None
    self.e2 = None
    self.r1 = None
    self.r2 = None

  def tensors(self):

    params = {'e1': randn(max(self.n,self.l),self.k),
              'e2': randn(max(self.n,self.l),self.k),
              'r1': randn(self.m,self.k),
              'r2': randn(self.m,self.k)}
    return params

  def lossfun(self):

    self.pred = T.sum(self.e1[self.rows,:] \
        * self.r1[self.cols,:] * self.e1[self.tubes,:], 1) \
      + T.sum(self.e2[self.rows,:] * self.r1[self.cols,:] \
        * self.e2[self.tubes,:], 1) \
      + T.sum(self.e1[self.rows,:] * self.r2[self.cols,:] \
        * self.e2[self.tubes,:], 1) \
      - T.sum(self.e2[self.rows,:] * self.r2[self.cols,:] \
        * self.e1[self.tubes,:], 1)

    self.loss = T.sqr(self.ys - self.pred).mean()

    self.regul = T.sqr(self.e1[self.rows,:]).mean() \
      + T.sqr(self.e2[self.rows,:]).mean() \
      + T.sqr(self.e1[self.tubes,:]).mean() \
      + T.sqr(self.e2[self.tubes,:]).mean() \
      + T.sqr(self.r1[self.cols,:]).mean() \
      + T.sqr(self.r2[self.cols,:]).mean()

  def objects(self, i, j):
    e1 = self.e1.get_value(borrow=True)
    r1 = self.r1.get_value(borrow=True)
    e2 = self.e2.get_value(borrow=True)
    r2 = self.r2.get_value(borrow=True)
    return ((e1[i,:] * r1[j,:]).dot(e1.T) + (e2[i,:] * r1[j,:]).dot(e2.T) + 
      (e1[i,:] * r2[j,:]).dot(e2.T) - (e2[i,:] * r2[j,:]).dot(e1.T))

  def subjects(self, j, k):
    e1 = self.e1.get_value(borrow=True)
    r1 = self.r1.get_value(borrow=True)
    e2 = self.e2.get_value(borrow=True)
    r2 = self.r2.get_value(borrow=True)
    return (e1.dot(r1[j,:] * e1[k,:]) + e2.dot(r1[j,:] * e2[k,:]) + 
      e1.dot(r2[j,:] * e2[k,:]) - e2.dot(r2[j,:] * e1[k,:]))

class RESCAL(Model):

  def __init__(self):
    super(RESCAL, self).__init__()
    self.name = self.__class__.__name__

    self.r = None
    self.e = None

  def tensors(self):

    params = {'r' : randn(self.m, self.k, self.k),
              'e' : randn(max(self.n,self.l),self.k)}
    return params

  def lossfun(self):

    self.pred = T.sum(T.sum(self.e[self.rows,:,None] * 
      self.r[self.cols,:,:], 1) *  self.e[self.tubes,:], 1)

    self.loss = T.sqr(self.ys - self.pred).mean()

    self.regul = T.sqr(self.e[self.rows,:]).mean() \
            + T.sqr(self.r[self.cols,:,:]).mean() \
            + T.sqr(self.e[self.tubes,:]).mean()

  def objects(self, i, j):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return (e[i,:].dot(r[j,:,:])).dot(e.T)

  def subjects(self, j, k):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return e.dot(r[j,:,:].dot(e[k,:]))

class DistMult(Model):

  def __init__(self):
    super(DistMult, self).__init__()
    self.name = self.__class__.__name__

    self.e = None
    self.r = None

  def tensors(self):
    params = {'e': randn(max(self.n, self.l), self.k),
              'r': randn(self.m, self.k)}
    return params

  def lossfun(self):

    self.pred = T.sum(self.e[self.rows,:] * self.r[self.cols,:] * 
      self.e[self.tubes,:], 1)

    self.loss = T.sqr(self.ys - self.pred).mean()

    self.regul = T.sqr(self.e[self.rows,:]).mean() \
      + T.sqr(self.r[self.cols,:]).mean() \
      + T.sqr(self.e[self.tubes,:]).mean()

  def objects(self, i, j):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return (e[i,:] * r[j,:]).dot(e.T)

  def subjects(self, j, k):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return e.dot(r[j,:] * e[k,:])

class TransE(Model):


  def __init__(self):
    super(TransE, self).__init__()
    self.name = self.__class__.__name__

    self.e = None
    self.r = None

  def tensors(self):
    params = {'e': randn(max(self.n, self.l), self.k),
              'r': L2(randn(self.m, self.k))}
    return params

  def lossfun(self):

    self.e.set_value(L2(self.e.get_value(borrow=True)), borrow=True)

    self.pred = 1 + T.sqrt(T.sum(T.sqr(self.e[self.rows,:] + 
      self.r[self.cols,:] - self.e[self.tubes,:]), 1))

    self.loss = T.sqr(self.ys - self.pred).mean()

    self.regul = 0

  def objects(self, i, j):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return - np.sum(np.square((e[i,:] + r[j,:]) - e), 1)

  def subjects(self, j, k):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return - np.sum(np.square(e + (r[j,:] - e[k,:])), 1)
