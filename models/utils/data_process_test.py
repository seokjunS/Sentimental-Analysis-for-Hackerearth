import os
import numpy as np
from data_preprocess import *


BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
DATA_PATH = os.path.join(BASE_PATH, 'data')



def num_of_unk(dataset):
  num_valid = 0
  num_pad = 0
  num_unk = 0

  for x in np.nditer(dataset.data['features']):
    if x == dataset.w2v_dict[PAD_WORD]:
      num_pad += 1
    elif x == dataset.w2v_dict[UNK_WORD]:
      num_unk += 1
    else:
      num_valid += 1

  print('Total words: %d, UNK: %d (%.3f), PAD: %d' % 
          (num_valid+num_unk, num_unk, float(num_unk)/(num_valid+num_unk), num_pad ) )








if __name__ == '__main__':
  ### setup
  w2v_dir = os.path.join(DATA_PATH, 'WV-500000')

  train_file = os.path.join(DATA_PATH, 'train_split.pkl')
  valid_file = os.path.join(DATA_PATH, 'val_split.pkl')

  batch_size = 12

  ### load w2v
  w2v_weights = np.load( os.path.join(w2v_dir, "weights.npy") )
  with open(os.path.join(w2v_dir, "dict.pkl"), 'r') as fr:
    w2v_dict = pickle.load(fr)
  
  ### make dataset
  train_set = Dataset(train_file, w2v_weights, w2v_dict, batch_size=batch_size)
  valid_set = Dataset(valid_file, w2v_weights, w2v_dict, batch_size=batch_size)


  # num_of_unk(train_set)
  # num_of_unk(valid_set)

  pass