from tools import *

class Batch(object):

  def __init__(self, positive, n_entities, bsize=100, neg_ratio=10):
    self.positive = positive
    self.bsize = bsize
    self.entities = n_entities
    self.neg_ratio = neg_ratio
    self.idx = 0

    self.new_triples_indexes = np.empty((self.bsize * (self.neg_ratio + 1) , 3)).astype(np.int64)
    self.new_triples_values = np.empty((self.bsize * (self.neg_ratio + 1 ))).astype(np.float32)

  def __call__(self):
    idxs = np.random.randint(0,len(self.positive.values),self.bsize)
    self.new_triples_indexes[:self.bsize,:] = self.positive.indexes[idxs,:]
    self.new_triples_values[:self.bsize] = self.positive.values[idxs]

    last_idx = self.bsize

    #Pre-sample everything, faster
    rdm_entities = np.random.randint(0, self.entities, last_idx * self.neg_ratio)
    rdm_choices = np.random.random(last_idx * self.neg_ratio)
    #Pre copying everyting
    self.new_triples_indexes[last_idx:(last_idx*(self.neg_ratio+1)),:] = np.tile(self.new_triples_indexes[:last_idx,:],(self.neg_ratio,1))
    self.new_triples_values[last_idx:(last_idx*(self.neg_ratio+1))] = np.tile(self.new_triples_values[:last_idx], self.neg_ratio)

    for i in range(last_idx):
      for j in range(self.neg_ratio):
        cur_idx = i* self.neg_ratio + j
        #Sample a random subject or object 
        if rdm_choices[cur_idx] < 0.5:
          self.new_triples_indexes[last_idx + cur_idx,0] = rdm_entities[cur_idx]
        else:
          self.new_triples_indexes[last_idx + cur_idx,2] = rdm_entities[cur_idx]

        self.new_triples_values[last_idx + cur_idx] = -1

    last_idx += cur_idx + 1

    train = [self.new_triples_values[:last_idx], self.new_triples_indexes[:last_idx,0], self.new_triples_indexes[:last_idx,1], self.new_triples_indexes[:last_idx,2]]


    return train



class TransE_Batch(Batch):
  #Hacky trick to normalize embeddings at each update
  def __init__(self, model, positive, entities, bsize=100, neg_ratio = 0.0):
    super(TransE_Batch, self).__init__(positive, entities, bsize, neg_ratio)

    self.model = model

  def __call__(self):
    train = super(TransE_Batch, self).__call__()
    train = train[1:]

    #Projection on L2 sphere before each batch
    self.model.e.set_value(L2(self.model.e.get_value(borrow = True)), borrow = True)

    return train
