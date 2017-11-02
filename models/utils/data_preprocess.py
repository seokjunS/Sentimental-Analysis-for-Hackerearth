import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import re


BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
DATA_PATH = os.path.join(BASE_PATH, 'data')


ENG_STOPS = set(stopwords.words("english"))
STEMMER = PorterStemmer()



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
" and split them in train vs. validation set (20%)
"""
def parse_training():
  print("[INFO] Start Cleansing train.csv")
  data = clean_csv( os.path.join(DATA_PATH, 'train.csv') )
  data.to_pickle( os.path.join(DATA_PATH, 'train.pkl') )

  print("[INFO] Start Splitting train.csv")
  # balancing labels
  pos_data = data[data['Is_Response'] == 1]
  neg_data = data[data['Is_Response'] == 0]

  train_pos_data = pos_data.sample(frac=0.8)
  train_neg_data = neg_data.sample(frac=0.8)

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
" DATASET class
"""
# class BKCullpdbDataset(object):
#   def __init__(self, ifile, window_size, split = "train", batch_size=BATCH_SIZE, with_profile = False, with_sol = False):
#     self.window_size = window_size
#     self.split = split
#     self.batch_size = batch_size
#     self.with_profile = with_profile
#     self.with_sol = with_sol

#     # get data
#     print("[%s: INFO] Start parsing %s cull pdb dataset" % (datetime.now(), split))
#     self.features, self.labels, self.sols = self.parse_data(ifile)
#     self.keys = ['features', 'labels']
#     self.data = {
#       'features': self.features,
#       'labels': self.labels,
#       'sols': self.sols
#     }

#     # make dataset
#     self.placeholders = {
#       'seed': tf.placeholder(tf.int64, shape=()),
#       'features': tf.placeholder(self.features.dtype, shape=self.features.shape),
#       'labels': tf.placeholder(self.labels.dtype, shape=self.labels.shape)
#     }

#     if with_sol:
#       self.keys.append( 'sols' )
#       self.placeholders['sols'] = tf.placeholder(self.sols.dtype, shape=self.sols.shape)

#     self.dataset = tf.contrib.data.Dataset.from_tensor_slices( tuple([ self.placeholders[k] for k in self.keys]) )\
#                       .shuffle(buffer_size=10000, seed=self.placeholders['seed'])\
#                       .batch(self.batch_size)
                                            
#     self.iterator = self.dataset.make_initializable_iterator()


#   def init(self, sess, seed=0):
#     print("[%s: INFO] Init dataset in TF" % (datetime.now()))
#     feed_dict = { self.placeholders[k]: self.data[k] for k in self.keys }
#     feed_dict[ self.placeholders['seed'] ] = seed * random.randrange(1,11)
#     sess.run(self.iterator.initializer, feed_dict=feed_dict)


#   def get_next(self):
#     return self.iterator.get_next()


#   def parse_data(self, ifile):
#     rawdata = np.load(ifile).astype(NP_DTYPE)
   
#     if self.split == "train":
#       data = rawdata[:5600]
#     elif self.split == "test":
#       data = rawdata[5600:5877]
#     elif self.split == "validation":
#       data = rawdata[5877:]
#     else:
#       print("[%s: ERROR] Invalid split type %s: " % (datetime.now(), self.split))
#       raise Exception


#     data = data.reshape(-1, 700, 57)[:1,:,:]

#     # amino acids
#     fidxs = list(range(0, 22))
    
#     if self.with_profile:
#       # profiles
#       fidxs.extend( list(range(35, 57)) )

#     features = data[:,:,fidxs]

#     labels = data[:,:,22:31]

#     sols = data[:,:,33:35]


#     ### set padding at features while '_' index to 1
#     padding = int( (self.window_size-1) / 2)
#     features = np.pad(features, ((0,0),(padding,padding),(0,0)), 'constant', constant_values=0.0)

#     features[:,:padding,AMINO_ACIDS['_']] = 1.0
#     features[:,-padding:,AMINO_ACIDS['_']] = 1.0

#     return features, labels, sols


#   def get_feature_shape(self):
#     return self.features.shape









if __name__ == '__main__':
  # parse_training()
  parse_test()

