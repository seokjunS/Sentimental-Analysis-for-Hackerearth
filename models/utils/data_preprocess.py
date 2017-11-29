import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
import os
import re
import gensim
from datetime import datetime
import tensorflow as tf
import random
import pickle


BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
DATA_PATH = os.path.join(BASE_PATH, 'data')


ENG_STOPS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

UNK_WORD = '_UNK'
PAD_WORD = '_PAD'


"""
" ETC
"""
def mkdir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)



"""
" Cleaning text
"""
def clean_data(text):
  # txt = str(text)
  txt = text
  txt = re.sub(r'[^A-Za-z0-9\s]',r'',txt)
  txt = re.sub(r'\n',r' ',txt)
  
  """
  1. lowercase
  2. remvoe stopwords
  3. stemming
  """
  filtered = []
  for w in txt.split():
    w.lower()
    if w not in ENG_STOPS:
      filtered.append( STEMMER.stem( w ) )

  txt = " ".join( filtered )
  return txt



"""
" Cleaning csv file
"""
def clean_csv(fname):
  data = pd.read_csv(fname)

  """
  1. tokenize 'description'
  2. keep 'User_ID', 'Description', 'Is_Response'(optional) => drop 'Browser_Used', 'Device_Used'
  """
  data['Description'] = data['Description'].map(lambda x: clean_data(x))
  print('data shaped', data.shape)
  data.drop('Browser_Used', axis=1, inplace=True)
  data.drop('Device_Used', axis=1, inplace=True)

  if 'Is_Response' in data.columns:
    data['Is_Response'] = [1 if x == 'happy' else 0 for x in data['Is_Response']]

  return data



"""
" Cleaning train.csv file
" and split them in train vs. validation set (10%)
"""
def parse_training():
  print("[INFO] Start Cleansing train.csv")
  data = clean_csv( os.path.join(DATA_PATH, 'train.csv') )
  data.to_pickle( os.path.join(DATA_PATH, 'train.pkl') )

  print("[INFO] Start Splitting train.csv")
  # balancing labels
  pos_data = data[data['Is_Response'] == 1]
  neg_data = data[data['Is_Response'] == 0]

  train_pos_data = pos_data.sample(frac=0.9)
  train_neg_data = neg_data.sample(frac=0.9)

  val_pos_data = pos_data.drop( train_pos_data.index )
  val_neg_data = neg_data.drop( train_neg_data.index )

  train_data = pd.concat([train_pos_data, train_neg_data]).reset_index(drop=True)
  val_data = pd.concat([val_pos_data, val_neg_data]).reset_index(drop=True)

  train_data.to_pickle( os.path.join(DATA_PATH, 'train_split.pkl') )
  val_data.to_pickle( os.path.join(DATA_PATH, 'val_split.pkl') )


def parse_training_split():
  print("[INFO] Start Cleansing train.csv")
  data = pd.read_csv( os.path.join(DATA_PATH, 'train.csv') )

  if 'Is_Response' in data.columns:
    data['Is_Response'] = [1 if x == 'happy' else 0 for x in data['Is_Response']]

  print("[INFO] Start Splitting train.csv")
  # balancing labels
  pos_data = data[data['Is_Response'] == 1]
  neg_data = data[data['Is_Response'] == 0]

  train_pos_data = pos_data.sample(frac=0.9)
  train_neg_data = neg_data.sample(frac=0.9)

  val_pos_data = pos_data.drop( train_pos_data.index )
  val_neg_data = neg_data.drop( train_neg_data.index )

  train_data = pd.concat([train_pos_data, train_neg_data]).reset_index(drop=True)
  val_data = pd.concat([val_pos_data, val_neg_data]).reset_index(drop=True)


  train_data = pd.concat([pd.Series(row['Is_Response'], sent_tokenize(row['Description'].decode('utf-8'))) \
                          for _, row in train_data.iterrows()]).reset_index()
  train_data.columns = ['Description', 'Is_Response']
  train_data['Description'] = train_data['Description'].map(lambda x: clean_data(x))
  
  
  # val_data = pd.concat([pd.Series(row['Is_Response'], sent_tokenize(row['Description'].decode('utf-8'))) \
  #                         for _, row in val_data.iterrows()]).reset_index()
  # val_data.columns = ['Description', 'Is_Response']
  # val_data['Description'] = val_data['Description'].map(lambda x: clean_data(x))

  train_data.to_pickle( os.path.join(DATA_PATH, 'train_split.pkl') )
  val_data.to_pickle( os.path.join(DATA_PATH, 'val_split.pkl') )




"""
" Cleaning test.csv file
"""
def parse_test():
  print("[INFO] Start Cleansing test.csv")
  data = clean_csv( os.path.join(DATA_PATH, 'test.csv') )
  data.to_pickle( os.path.join(DATA_PATH, 'test.pkl') )



"""
" Trim word2vec
"""
def trim_word2vec(size):
  pretrained = os.path.join(DATA_PATH, 'GoogleNews-vectors-negative300.bin')
  outpath = os.path.join(DATA_PATH, "WV-%d"%size)
  mkdir(outpath)
  w_fname = os.path.join(outpath, 'weights.npy')
  d_fname = os.path.join(outpath, 'dict.pkl')


  model = gensim.models.KeyedVectors.load_word2vec_format(pretrained, binary=True, limit=size+1)

  w2v = np.zeros((size+2,300))

  # with open(d_fname, 'w') as dict_file:
  #   for i, word in enumerate(model.wv.index2word): # i: 0 ~ size
  #     w2v[i] = model[word]
  #     if i == size: # last
  #       dict_file.write("%s\n"%UNK_WORD.encode('utf-8'))
  #     else:
  #       dict_file.write("%s\n"%word.encode('utf-8'))
  #   # last index for padding
  #   dict_file.write("%s\n"%PAD_WORD.encode('utf-8'))
  w2v_dict = dict()
  
  for i, word in enumerate(model.wv.index2word): # i: 0 ~ size
    w2v[i] = model[word]
    if i == size: # last
      w2v_dict[ UNK_WORD.encode('utf-8') ] = i
    else:
      w2v_dict[ word.encode('utf-8') ] = i
  # last index for padding
  w2v_dict[ PAD_WORD.encode('utf-8') ] = size+1


  np.save(w_fname, w2v)
  with open(d_fname, 'w') as dict_file:
    pickle.dump(w2v_dict, dict_file)



"""
parse fasttext vec file
"""
def parse_fasttext():
  base_path = os.path.join(DATA_PATH, 'fasttext')
  input_path = os.path.join(base_path, 'combined.vec')
  dict_file = os.path.join(base_path, 'dict.pkl')
  weight_file = os.path.join(base_path, 'weights.npy')

  # merge last 

  with open(input_path, 'r') as fr:
    dsize, flen = fr.readline().rstrip().split(" ")
    dsize = int(dsize)
    flen = int(flen)

    # make last word as UNK_WORD and all zero weight for PAD
    w2v_dict = dict()
    w2v = np.zeros((dsize+1, flen))

    for i, line in enumerate(fr.readlines()):
      line = line.rstrip().split(" ")
      word = line[0]
      features = [ float(n) for n in line[1:] ]

      w2v[i] = features
      if i == dsize - 1:
        w2v_dict[ UNK_WORD ] = i
      else:
        w2v_dict[ word ] = i

    # PAD_WORD
    w2v_dict[ PAD_WORD ] = dsize

  np.save(weight_file, w2v)
  with open(dict_file, 'w') as fw:
    pickle.dump(w2v_dict, fw)





"""
" DATASET class
"""
class Dataset(object):
  def __init__(self, datafile, w2v_weights, w2v_dict, batch_size, max_epoch=1, need_shuffle=True):
    self.batch_size = batch_size
    self.w2v_weights = w2v_weights
    self.w2v_dict = w2v_dict
    self.max_epoch = max_epoch
    self.need_shuffle = need_shuffle

    ### get data
    rawdata = pd.read_pickle(datafile)

    ### make word to index
    self.num_data = rawdata.shape[0]
    maxlen = 0
    for idx, row in rawdata.iterrows():
      l = len(row['Description'].split())
      maxlen = max(maxlen, l)

    self.data = np.zeros((self.num_data, maxlen))
    self.labels = np.zeros((self.num_data)) if 'Is_Response' in rawdata.columns else None

    for i, row in rawdata.iterrows():
      words = row['Description'].split()
      slen = len(words)

      for j in range(maxlen):
        if j < slen:
          self.data[i,j] = self.encode_word( words[j] )
        else:
          self.data[i,j] = self.w2v_dict[PAD_WORD]

      if self.labels is not None:
        self.labels[i] = row['Is_Response']

    ### batching
    self.perm_idxs = np.arange(self.num_data)

    self.epochs_completed = 0
    self.index_in_epoch = 0

    self.init()
    


  def encode_word(self, w):
    default = self.w2v_dict[UNK_WORD]
    return self.w2v_dict.get( w, default )


  def init(self):
    if self.need_shuffle:
      np.random.shuffle(self.perm_idxs)
    self.index_in_epoch = 0

  def reset(self):
    self.epochs_completed = 0
    self.init()


  """
  Get next batch.
  If next element is not available, return None.
  """
  def next_batch(self):
    start = self.index_in_epoch
    self.index_in_epoch += self.batch_size
    end = self.index_in_epoch
    epoch_finish = False

    # if end index overflow
    if end > self.num_data:
      end = self.num_data
      epoch_finish = True

    # if next batch is empty
    if end == self.num_data:
      epoch_finish = True
      
    size = end - start

    if size == 0:
      return None, None, epoch_finish
    else:
      idxs = self.perm_idxs[start:end]
      return self.data[idxs], self.labels[idxs], epoch_finish


  def iter_batch(self):
    while self.epochs_completed < self.max_epoch:
      data, labels, epoch_finish = self.next_batch()
      cnt_epoch = None

      if epoch_finish:
        print("[INFO] Epoch %d finished!" % self.epochs_completed)
        cnt_epoch = self.epochs_completed
        self.epochs_completed += 1
        self.init()

      if data is None:
        # print("[DEBUG] reach here")
        continue

      yield data, labels, cnt_epoch



  def get_feature_shape(self):
    return self.data.shape




"""
" DATASET class
" for validation, so each batch is one data
"""
class ValidDataset(object):
  def __init__(self, datafile, w2v_weights, w2v_dict):
    self.w2v_weights = w2v_weights
    self.w2v_dict = w2v_dict

    ### get data
    rawdata = pd.read_pickle(datafile)

    ### make word to index
    self.num_data = rawdata.shape[0]

    self.data = []
    self.labels = np.zeros((self.num_data)) if 'Is_Response' in rawdata.columns else None

    for idx, row in rawdata.iterrows():
      maxlen = 0
      sentences = sent_tokenize(row['Description'].decode('utf-8'))
      sentences = [ clean_data(s) for s in sentences ]

      for idx, s in enumerate(sentences):
        l = len(s.split())
        maxlen = max(maxlen, l)

      data = np.zeros((len(sentences), maxlen))

      for i, s in enumerate(sentences):
        words = s.split()
        slen = len(words)

        for j in range(maxlen):
          if j < slen:
            data[i,j] = self.encode_word( words[j] )
          else:
            data[i,j] = self.w2v_dict[PAD_WORD]


      self.data.append( data )
      if self.labels is not None:
        self.labels[i] = row['Is_Response']


  def encode_word(self, w):
    default = self.w2v_dict[UNK_WORD]
    return self.w2v_dict.get( w, default )


  def iter(self):
    for idx, data in enumerate(self.data):
      yield data, [self.labels[idx]]








# """
# " DATASET class
# """
# class Dataset(object):
#   def __init__(self, datafile, w2v_weights, w2v_dict, batch_size):
#     self.batch_size = batch_size
#     self.w2v_weights = w2v_weights
#     self.w2v_dict = w2v_dict

#     ### get data
#     rawdata = pd.read_pickle(datafile)

#     ### make word to index
#     origin_num = rawdata.shape[0]
#     self.maxlen = 1000
#     for idx, row in rawdata.iterrows():
#       lst = row['Description'].split()
#       l = len(lst)
#       if l > self.maxlen:
#         rawdata.drop(idx, inplace=True)
#     # rawdata = rawdata[ len(rawdata['Description'].split()) <= self.maxlen ]

#     print("%d data filtered" % (origin_num - rawdata.shape[0]))


#     self.data = {
#       'features': np.zeros((rawdata.shape[0], self.maxlen)),
#       # 'labels': np.zeros((self.num_valid)),
#       'labels': rawdata['Is_Response'].values if 'Is_Response' in rawdata.columns else None,
#       'ids': rawdata['User_ID'].values
#     }


#     for i, (_, row) in enumerate(rawdata.iterrows()):
#       split = row['Description'].split()
#       slen = len(split)
#       for j in range(self.maxlen):
#         if j < slen:
#           self.data['features'][i,j] = self.encode_word( split[j] )
#         else:
#           self.data['features'][i,j] = self.w2v_dict[PAD_WORD]


#     ### make dataset
#     self.seed = tf.placeholder(tf.int64, shape=(), name='dataset_seed_ph')
#     self.placeholders = {
#       'features': tf.placeholder(tf.int32, shape=self.data['features'].shape, name='dataset_feature_ph'),
#       'labels': tf.placeholder(tf.int32, shape=self.data['labels'].shape, name='dataset_label_ph')
#     }

#     self.dataset = tf.contrib.data.Dataset.from_tensor_slices( tuple([ self.placeholders[k] for k in self.placeholders]) )\
#                       .shuffle(buffer_size=1000, seed=self.seed)\
#                       .batch(self.batch_size)
                                            
#     self.iterator = self.dataset.make_initializable_iterator()


#   def encode_word(self, w):
#     default = self.w2v_dict[UNK_WORD]
#     return self.w2v_dict.get( w, default )


#   def init(self, sess, seed=0):
#     print("[%s: INFO] Init dataset in TF" % (datetime.now()))
#     feed_dict = { self.placeholders[k]: self.data[k] for k in self.placeholders }
#     feed_dict[ self.seed ] = seed * random.randrange(1,11)
#     sess.run(self.iterator.initializer, feed_dict=feed_dict)


#   def get_next(self):
#     return self.iterator.get_next()


#   def get_feature_shape(self):
#     return self.data['features'].shape




if __name__ == '__main__':
  parse_training()
  # parse_training_split()
  # parse_test()
  # trim_word2vec(50000)
  # parse_fasttext()

  pass

