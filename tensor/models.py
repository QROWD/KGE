import downhill
import theano
import theano.tensor as TT

from batching import * 
from evaluation import *
from log import *
from tools import *

theano.config.floatX = 'float32'
theano.config.mode = 'FAST_RUN'
theano.config.exception_verbosity = 'high'

class Model(object):

  def __init__(self):
    self.name = self.__class__.__name__
    
    #Symbolic expressions for the prediction function (and compiled one too), the loss, the regularization, and the loss to
    #optimize (loss + lmbda * regul)
    #To be defined by the child classes:
    self.pred_func = None
    self.pred_func_compiled = None

    self.loss_func = None
    self.regul_func = None
    self.loss_to_opt = None
    
    #Symbolic variables for training values
    self.ys = TT.vector('ys')
    self.rows = TT.lvector('rows')
    self.cols = TT.lvector('cols')
    self.tubes = TT.lvector('tubes') 


    #Current values for which the loss is currently compiled
    #3 dimensions:
    self.n = 0 #Number of subject entities
    self.m = 0 #Number of relations
    self.l = 0 #Number of object entities
    #and rank:
    self.k = 0
    #and corresponding number of parameters (i.e. n*k + m*k + l*k for CP_Model)
    self.nb_params = 0



  def set_dims(self,train_triples, hparams):
    self.n = max(train_triples.indexes[:,0]) +1
    self.m = max(train_triples.indexes[:,1]) +1
    self.l = max(train_triples.indexes[:,2]) +1
    self.k = hparams.embedding_size


  def get_pred_symb_vars(self):
    """
    Returns the default pred parameters. Made to be overriden by child classes if need of additional input variables
    """
    pred_inputs=[self.rows, self.cols, self.tubes]

    return pred_inputs

  def get_pred_args(self,test_idxs):
    """
    Returns the default pred symbolic variables. Made to be overriden by child classes if need of additional input variables
    """

    return [test_idxs[:,0], test_idxs[:,1], test_idxs[:,2]]


  def get_loss_args_and_symb_vars(self, train_triples, valid_triples, hparams):
    """
    Returns the default loss parameters and corresponding symbolic variables. Made to be overriden by child classes if need of additional input variables
    """

    train = Batch_Loader(train_triples, n_entities = max(self.n,self.l), batch_size = hparams.batch_size, neg_ratio = hparams.neg_ratio, contiguous_sampling = False ) 
    inputs=[self.ys, self.rows, self.cols, self.tubes]
    if valid_triples != None:
      valid = Batch_Loader(valid_triples, n_entities = max(self.n,self.l), batch_size = len(valid_triples.values), neg_ratio = hparams.neg_ratio, contiguous_sampling = False ) 
      #valid = [valid_triples.values[:], valid_triples.indexes[:,0], valid_triples.indexes[:,1], valid_triples.indexes[:,2]]
    else:
      valid = None

    return train, inputs, valid

  def allocate_params(self):
    nb_params=0
    #Call child class getter of initial values of model parameters:
    params = self.get_init_params()
    #And allocate them as theano shared variables
    for name, val in params.items():
      setattr(self, name, theano.shared(val, name = name) )
      nb_params += val.size
    self.nb_params = nb_params

  def reinit_params(self):
    nb_params=0
    #Call child class getter of initial values of model parameters:
    params = self.get_init_params()
    #And set their values
    for name, val in params.items():
      getattr(self, name).set_value(val, borrow  = True)
      nb_params += val.size
    self.nb_params = nb_params


  def setup_params_for_train(self,train_triples, valid_triples, hparams, redefine_loss = False):
    """
    Calls teh define_loss function that gives the model loss to minimize
    """


    #Check if need to redefine the loss function or not:
    if redefine_loss or self.loss_to_opt == None:

      #Initialize parameters (child class overriding):
      self.allocate_params()

      #Defining the model (child class overriding):
      self.define_loss()

      #Compile the prediction functions:
      self.pred_func_compiled = theano.function(self.get_pred_symb_vars(), self.pred_func)

    else:
      #Just reinit the params
      self.reinit_params()

    #Combine loss and regularization to get what downhill will optimize:
    #Changing the scalar value lmbda in the function actually doesn't make theano to recompile everything, it's as fast as not changing anything
    self.loss_to_opt = self.loss + hparams.lmbda * self.regul_func


  def fit(self, train_triples, valid_triples, hparams, n=0,m=0,l=0, scorer = None):

    #Set input_dimensions:
    if n == 0: #No given dimensions, can be useful for transparent predicton of entities/rels not seen in train
      self.set_dims(train_triples, hparams)
    else:
      self.n, self.m, self.l, self.k = n, m, l, hparams.embedding_size

    #Define the downhill loss corresponding to the input dimensions
    self.setup_params_for_train(train_triples, valid_triples, hparams)
    
    #get the loss inputs:
    train_vals, train_symbs, valid_vals = self.get_loss_args_and_symb_vars(train_triples, valid_triples, hparams)

    opt = downhill.build(hparams.learning_policy, loss=self.loss_to_opt, inputs=train_symbs, monitor_gradients=True)

    train_vals = downhill.Dataset(train_vals, name = 'train')


    #Main SGD loop
    it = 1
    best_valid_mrr = -1
    best_valid_ap = -1
    for tm, vm in opt.iterate(train_vals, None,
        max_updates=hparams.max_iter,
        validate_every=9999999,         #I take care of the valiation, with validation metrics instead of loss
        patience=9999999,           #Number of tolerated imporvements of validation loss that are inferior to min_improvement
        max_gradient_norm=1,                # Prevent gradient explosion!
        learning_rate=hparams.learning_rate):


      if it % hparams.valid_scores == 0 and scorer != None:
        if valid_triples != None:
          logger.info("Validation metrics:")
          res = scorer.compute(self, valid_triples)
          cv_res = Results()
          cv_res.add(res)


          metrics = cv_res.measures()
            
          #Early stopping on filtered MRR
          if best_valid_mrr >= metrics[0]:
            logger.info("Validation filtered MRR decreased, stopping here.")
            break
          else:
            best_valid_mrr = metrics[0]
        else:
          logger.info("Validation AP: " + str(res.ap))
          #Early stopping on Average Precision
          if best_valid_ap >= res.ap:
            logger.info("Validation AP decreased, stopping here.")
            break
          else:
            best_valid_ap = res.ap

      it += 1
      if it >= hparams.max_iter: #Avoid downhill resetting the parameters when max_iter is reached
        break



  def predict(self, test_idxs):
    """
    test_idxs is a 2D numpy array of size [a,3], containing the indexes of the test triples.
    Returns a vector of size a.
    """

    return self.pred_func_compiled( *self.get_pred_args(test_idxs) )


################ Abstract Functions: ('must' be defined by child classes)


  def get_init_params(self,n,m,l,k):
    """
    Abstract. Returns a dict of the initial values of shared variables, indexes by the class attribute name (string).
    """
    pass



  def define_loss(self):
    """
    Abstract. Define the loss of the model in the child class for a given input (theano shared variables
    must have the correct dimensions at loss definition). Implies an initialization of the parameters
    (shared variables), and the initialization of three functions (prediction, loss and regularization) 
    as symbolic theano expressions: self.pred_func, self.loss_func and self.regul_func (mandatory).
    """
    pass





#################################################################################################
# Definition of models (child classes):
#################################################################################################





class CP_Model(Model):
  """
  Canonical Polyadic decomposition model
  """

  def __init__(self):
    super(CP_Model, self).__init__()
    
    self.name = self.__class__.__name__
    
    #Parameters: 3 embeddings matrices
    #Will be initialized at train time
    self.u = None
    self.v = None
    self.w = None


  def get_init_params(self):

    params = { 'u' : randn(max(self.n,self.l),self.k),
           'v' : randn(self.m,self.k),
           'w' : randn(max(self.n,self.l),self.k)  }
    return params


  def define_loss(self):

    #Defining the loss by decomposing each part:
    self.pred_func = TT.sum(self.u[self.rows,:] * self.v[self.cols,:] * self.w[self.tubes,:], 1)

    self.loss = TT.sqr(self.ys - self.pred_func).mean()

    self.regul_func = TT.sqr(self.u[self.rows,:]).mean() \
            + TT.sqr(self.v[self.cols,:]).mean() \
            + TT.sqr(self.w[self.tubes,:]).mean()

  def eval_o(self, i, j):
    u = self.u.get_value(borrow=True)
    v = self.v.get_value(borrow=True)
    w = self.w.get_value(borrow=True)
    return (u[i,:] * v[j,:]).dot(w.T)

  def eval_s(self, j, k):
    u = self.u.get_value(borrow=True)
    v = self.v.get_value(borrow=True)
    w = self.w.get_value(borrow=True)
    return u.dot(v[j,:] * w[k,:])


class CP_Logistic(CP_Model):
  """
  Canonical Polyadic decomposition model with logistic loss. Inherit from CP_Model
  """

  def __init__(self):
    super(CP_Logistic, self).__init__()
    
    self.name = self.__class__.__name__
    

  def define_loss(self):
    #Defining the loss by decomposing each part:
    self.pred_func = TT.nnet.sigmoid( TT.sum(self.u[self.rows,:] * self.v[self.cols,:] * self.w[self.tubes,:], 1))

    self.loss = TT.nnet.softplus( - self.ys * TT.sum(self.u[self.rows,:] * self.v[self.cols,:] * self.w[self.tubes,:], 1)).mean()

    self.regul_func = TT.sqr(self.u[self.rows,:]).mean() \
            + TT.sqr(self.v[self.cols,:]).mean() \
            + TT.sqr(self.w[self.tubes,:]).mean()






class DistMult_Model(Model):
  """
  DistMult model
  """

  def __init__(self):
    super(DistMult_Model, self).__init__()
    
    self.name = self.__class__.__name__
    
    self.e = None
    self.r = None


  def get_init_params(self):

    params = { 'e' : randn(max(self.n,self.l),self.k),
           'r' : randn(self.m,self.k)}
    return params


  def define_loss(self):

    self.pred_func = TT.sum(self.e[self.rows,:] * self.r[self.cols,:] * self.e[self.tubes,:], 1)

    self.loss = TT.sqr(self.ys - self.pred_func).mean()

    self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
            + TT.sqr(self.r[self.cols,:]).mean() \
            + TT.sqr(self.e[self.tubes,:]).mean()

  def eval_o(self, i, j):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return (e[i,:] * r[j,:]).dot(e.T)

  def eval_s(self, j, k):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return e.dot(r[j,:] * e[k,:])


class DistMult_Logistic(DistMult_Model):
  """
  DistMult model with logistic loss
  """

  def __init__(self):
    super(DistMult_Logistic, self).__init__()
    
    self.name = self.__class__.__name__
    

  def define_loss(self):
    self.pred_func = TT.nnet.sigmoid( TT.sum(self.e[self.rows,:] * self.r[self.cols,:] * self.e[self.tubes,:], 1))

    self.loss = TT.nnet.softplus( - self.ys * TT.sum(self.e[self.rows,:] * self.r[self.cols,:] * self.e[self.tubes,:], 1)).mean()

    self.regul_func = TT.sqr(self.e[self.rows,:]).mean() \
            + TT.sqr(self.r[self.cols,:]).mean() \
            + TT.sqr(self.e[self.tubes,:]).mean()




class TransE_L2(Model):
  """
  TransE model with L2 loss, but with the max margin ranking loss (performs better than maximum likelihood, only for TransE interestingly).
  """

  def __init__(self):
    super(TransE_L2, self).__init__()
    
    self.name = self.__class__.__name__
    
    self.e = None
    self.r = None

    #Need to split the data
    self.batch_size = None
    self.neg_ratio = None


  def get_init_params(self):

    params = { 'e' : randn(max(self.n,self.l),self.k), #np.random.uniform(-6.0/np.sqrt(self.k), 6.0/np.sqrt(self.k), (max(self.n,self.l),self.k)).astype('f')
           'r' : L2_proj(randn(self.m,self.k)), #L2_proj(np.random.uniform(-6.0/np.sqrt(self.k), 6.0/np.sqrt(self.k), (self.m,self.k))).astype('f') 
           }
    return params

  def get_loss_args_and_symb_vars(self, train_triples, valid_triples, hparams):
    """
    Override, regularization made in the batch loader (hacky)
    """

    train = TransE_Batch_Loader(self, train_triples, n_entities = max(self.n,self.l), batch_size = hparams.batch_size,
         neg_ratio = hparams.neg_ratio, contiguous_sampling = False)  
    inputs=[self.rows, self.cols, self.tubes]
    if valid_triples != None:
      valid = Batch_Loader(valid_triples, n_entities = max(self.n,self.l), batch_size = len(valid_triples.values), 
          neg_ratio = hparams.neg_ratio, contiguous_sampling = False)    
    else:
      valid = None

    return train, inputs, valid


  def setup_params_for_train(self, train_triples, valid_triples, hparams, redefine_loss = False):
    self.batch_size = hparams.batch_size
    self.neg_ratio = float(hparams.neg_ratio)

    self.margin = hparams.lmbda #Use for cross-validation since there is no need of regularization

    super(TransE_L2,self).setup_params_for_train(train_triples, valid_triples, hparams, redefine_loss)



  def define_loss(self):

    self.pred_func = - TT.sqrt(TT.sum(TT.sqr(self.e[self.rows,:] + self.r[self.cols,:] - self.e[self.tubes,:]),1))

    self.loss = TT.maximum( 0, self.margin + TT.sqrt(TT.sum(TT.sqr(self.e[self.rows[:self.batch_size],:] + self.r[self.cols[:self.batch_size],:] - self.e[self.tubes[:self.batch_size],:]),1) ) \
      - (1.0/self.neg_ratio) * TT.sum(TT.sqrt(TT.sum(TT.sqr(self.e[self.rows[self.batch_size:],:] + self.r[self.cols[self.batch_size:],:] - self.e[self.tubes[self.batch_size:],:]),1)).reshape((int(self.batch_size),int(self.neg_ratio))),1) ).mean()

    #Maximum likelihood loss, performs worse.
    #self.loss = TT.nnet.softplus( self.ys * TT.sqrt(TT.sum(TT.sqr(self.e[self.rows,:] + self.r[self.cols,:] - self.e[self.tubes,:]),1)) ).mean()

    self.regul_func = 0 #Entities embeddings are force to have unit norm, as stated in TransE paper

  def eval_o(self, i, j):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return - np.sum(np.square((e[i,:] + r[j,:]) - e ),1)

  def eval_s(self, j, k):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return - np.sum(np.square(e + (r[j,:] - e[k,:]) ),1)


class TransE_L1(TransE_L2):
  """
  TransE model with L1 loss, but with the max margin ranking loss (performs better than maximum likelihood, only for TransE interestingly).
  """

  def __init__(self):
    super(TransE_L1, self).__init__()
    
    self.name = self.__class__.__name__
    


  def setup_params_for_train(self, train_triples, valid_triples, hparams, redefine_loss = False):

    super(TransE_L1,self).setup_params_for_train(train_triples, valid_triples, hparams, redefine_loss)

  def define_loss(self):

    self.pred_func = - TT.sum(TT.abs_(self.e[self.rows,:] + self.r[self.cols,:] - self.e[self.tubes,:]),1)

    self.loss = TT.maximum( 0, self.margin + TT.sum(TT.abs_(self.e[self.rows[:self.batch_size],:] + self.r[self.cols[:self.batch_size],:] - self.e[self.tubes[:self.batch_size],:]),1) \
      - (1.0/self.neg_ratio) * TT.sum(TT.sum(TT.abs_(self.e[self.rows[self.batch_size:],:] + self.r[self.cols[self.batch_size:],:] - self.e[self.tubes[self.batch_size:],:]),1).reshape((int(self.batch_size),int(self.neg_ratio))),1) ).mean()

    self.regul_func = 0 

  def eval_o(self, i, j):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return - np.sum(np.abs((e[i,:] + r[j,:]) - e),1)

  def eval_s(self, j, k):
    e = self.e.get_value(borrow=True)
    r = self.r.get_value(borrow=True)
    return - np.sum(np.abs(e + (r[j,:] - e[k,:])),1)


class Complex_Model(Model):
  """
  Factorization in complex numbers
  """

  def __init__(self):
    super(Complex_Model, self).__init__()
    
    self.name = self.__class__.__name__
    
    #Parameters:
    self.e1 = None
    self.e2 = None
    self.r1 = None
    self.r2 = None


  def get_init_params(self):

    params = { 'e1' : randn(max(self.n,self.l),self.k),
           'e2' : randn(max(self.n,self.l),self.k),
           'r1' : randn(self.m,self.k),
           'r2' : randn(self.m,self.k)  }
    return params


  def define_loss(self):

    self.pred_func = TT.sum(self.e1[self.rows,:] * self.r1[self.cols,:] * self.e1[self.tubes,:], 1) \
             + TT.sum(self.e2[self.rows,:] * self.r1[self.cols,:] * self.e2[self.tubes,:], 1) \
             + TT.sum(self.e1[self.rows,:] * self.r2[self.cols,:] * self.e2[self.tubes,:], 1) \
             - TT.sum(self.e2[self.rows,:] * self.r2[self.cols,:] * self.e1[self.tubes,:], 1) 

    self.loss = TT.sqr(self.ys - self.pred_func).mean()

    self.regul_func = TT.sqr(self.e1[self.rows,:]).mean() \
            + TT.sqr(self.e2[self.rows,:]).mean() \
            + TT.sqr(self.e1[self.tubes,:]).mean() \
            + TT.sqr(self.e2[self.tubes,:]).mean() \
            + TT.sqr(self.r1[self.cols,:]).mean() \
            + TT.sqr(self.r2[self.cols,:]).mean()


class Complex_Logistic(Complex_Model):
  """
  Factorization in complex numbers with logistic loss
  """

  def __init__(self):
    super(Complex_Logistic, self).__init__()
    self.name = self.__class__.__name__
    


  def define_loss(self):

    #Inverse since those that have a smaller distance are the most probable.
    self.pred_func =  TT.nnet.sigmoid( TT.sum(self.e1[self.rows,:] * self.r1[self.cols,:] * self.e1[self.tubes,:], 1) \
             + TT.sum(self.e2[self.rows,:] * self.r1[self.cols,:] * self.e2[self.tubes,:], 1) \
             + TT.sum(self.e1[self.rows,:] * self.r2[self.cols,:] * self.e2[self.tubes,:], 1) \
             - TT.sum(self.e2[self.rows,:] * self.r2[self.cols,:] * self.e1[self.tubes,:], 1) )

    self.loss = TT.nnet.softplus( - self.ys * ( TT.sum(self.e1[self.rows,:] * self.r1[self.cols,:] * self.e1[self.tubes,:], 1) \
             + TT.sum(self.e2[self.rows,:] * self.r1[self.cols,:] * self.e2[self.tubes,:], 1) \
             + TT.sum(self.e1[self.rows,:] * self.r2[self.cols,:] * self.e2[self.tubes,:], 1) \
             - TT.sum(self.e2[self.rows,:] * self.r2[self.cols,:] * self.e1[self.tubes,:], 1) )).mean()

    self.regul_func = TT.sqr(self.e1[self.rows,:]).mean() \
            + TT.sqr(self.e2[self.rows,:]).mean() \
            + TT.sqr(self.e1[self.tubes,:]).mean() \
            + TT.sqr(self.e2[self.tubes,:]).mean() \
            + TT.sqr(self.r1[self.cols,:]).mean() \
            + TT.sqr(self.r2[self.cols,:]).mean()


  def eval_o(self, i, j):
    e1 = self.e1.get_value(borrow=True)
    r1 = self.r1.get_value(borrow=True)
    e2 = self.e2.get_value(borrow=True)
    r2 = self.r2.get_value(borrow=True)
    return (e1[i,:] * r1[j,:]).dot(e1.T) + (e2[i,:] * r1[j,:]).dot(e2.T) + (e1[i,:] * r2[j,:]).dot(e2.T) - (e2[i,:] * r2[j,:]).dot(e1.T)

  def eval_s(self, j, k):
    e1 = self.e1.get_value(borrow=True)
    r1 = self.r1.get_value(borrow=True)
    e2 = self.e2.get_value(borrow=True)
    r2 = self.r2.get_value(borrow=True)
    return e1.dot(r1[j,:] * e1[k,:]) + e2.dot(r1[j,:] * e2[k,:]) + e1.dot(r2[j,:] * e2[k,:]) - e2.dot(r2[j,:] * e1[k,:])


