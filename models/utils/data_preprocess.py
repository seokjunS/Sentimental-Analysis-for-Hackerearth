import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
  txt = str(text)
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
" DATASET class
"""
class Dataset(object):
  def __init__(self, datafile, w2v_weights, w2v_dict, batch_size):
    self.batch_size = batch_size
    self.w2v_weights = w2v_weights
    self.w2v_dict = w2v_dict

    ### get data
    rawdata = pd.read_pickle(datafile)

    ### make word to index
    origin_num = rawdata.shape[0]
    self.maxlen = 1000
    for idx, row in rawdata.iterrows():
      lst = row['Description'].split()
      l = len(lst)
      if l > self.maxlen:
        rawdata.drop(idx, inplace=True)
    # rawdata = rawdata[ len(rawdata['Description'].split()) <= self.maxlen ]

    print("%d data filtered" % (origin_num - rawdata.shape[0]))


    self.data = {
      'features': np.zeros((rawdata.shape[0], self.maxlen)),
      # 'labels': np.zeros((self.num_valid)),
      'labels': rawdata['Is_Response'].values if 'Is_Response' in rawdata.columns else None,
      'ids': rawdata['User_ID'].values
    }


    for i, (_, row) in enumerate(rawdata.iterrows()):
      split = row['Description'].split()
      slen = len(split)
      for j in range(self.maxlen):
        if j < slen:
          self.data['features'][i,j] = self.encode_word( split[j] )
        else:
          self.data['features'][i,j] = self.w2v_dict[PAD_WORD]


    ### make dataset
    self.seed = tf.placeholder(tf.int64, shape=(), name='dataset_seed_ph')
    self.placeholders = {
      'features': tf.placeholder(tf.int32, shape=self.data['features'].shape, name='dataset_feature_ph'),
      'labels': tf.placeholder(tf.int32, shape=self.data['labels'].shape, name='dataset_label_ph')
    }

    self.dataset = tf.contrib.data.Dataset.from_tensor_slices( tuple([ self.placeholders[k] for k in self.placeholders]) )\
                      .shuffle(buffer_size=1000, seed=self.seed)\
                      .batch(self.batch_size)
                                            
    self.iterator = self.dataset.make_initializable_iterator()


  def encode_word(self, w):
    default = self.w2v_dict[UNK_WORD]
    return self.w2v_dict.get( w, default )


  def init(self, sess, seed=0):
    print("[%s: INFO] Init dataset in TF" % (datetime.now()))
    feed_dict = { self.placeholders[k]: self.data[k] for k in self.placeholders }
    feed_dict[ self.seed ] = seed * random.randrange(1,11)
    sess.run(self.iterator.initializer, feed_dict=feed_dict)


  def get_next(self):
    return self.iterator.get_next()


  def get_feature_shape(self):
    return self.data['features'].shape




if __name__ == '__main__':
  # parse_training()
  # parse_test()
  # trim_word2vec(500000)

  pass

