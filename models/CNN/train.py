import sys
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse
import time
import random
import pickle
from sklearn.metrics import accuracy_score

sys.path.append( os.path.join(os.path.dirname(__file__), "..") )
from utils.data_preprocess import Dataset
from model import *


BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
DATA_PATH = os.path.join(BASE_PATH, 'data')
TRAIN_PATH = os.path.join(BASE_PATH, 'trained')

"""
Set parameters
"""
def arg_parse(args):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--w2v_dir',
      type=str,
      # default=os.path.join(DATA_PATH, 'WV-500000'),
      default=os.path.join(DATA_PATH, 'WV-50000'),
      help='Word2Vec path'
  )
  parser.add_argument(
      '--tune_embedding',
      type=bool,
      default=False,
      help='tune_embedding'
  )
  parser.add_argument(
      '--keep_prob',
      type=float,
      default=0.7,
      help='Rate to be kept for dropout.'
  )
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.005,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_epoch',
      type=int,
      default=100,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--train_file',
      type=str,
      default=os.path.join(DATA_PATH, 'train_split.pkl'),
      help='Directory for input data.'
  )
  parser.add_argument(
      '--val_file',
      type=str,
      default=os.path.join(DATA_PATH, 'val_split.pkl'),
      help='Directory for input data.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='',
      help='Path to write log, checkpoint, and summary files.'
  )
  parser.add_argument(
      '--validation_capacity',
      type=int,
      default=10,
      help='How many validations should be considered for early stopping (--validation_capacity N means considering previous N-1 epochs).'
  )
  parser.add_argument(
      '--log_interval',
      type=int,
      default=100,
      help='Interval of steps for logging.'
  )
  parser.add_argument(
      '--validation_interval',
      type=int,
      default=1,
      help='Interval of steps for logging.'
  )
  parser.add_argument(
      '--filter_width',
      type=int,
      nargs='+',
      default=[5],
      help='Space seperated list of filter widths. (ex, --filter_width 4 8 12)'
  )
  parser.add_argument(
      '--num_filters',
      type=int,
      nargs='+',
      default=[300],
      help='Space seperated list of number of filters. (ex, --num_filters 4 8 12)'
  )
  parser.add_argument(
      '--l2_reg',
      type=float,
      default=0.0001,
      help=''
  )
  parser.add_argument(
      '--do_max_pool',
      type=bool,
      default=False,
      help=''
  )

  FLAGS, unparsed = parser.parse_known_args(args)

  return FLAGS, unparsed





"""
class Monitor
"""
class ValidationMonitor:
  def __init__(self, capacity):
    self.capacity = capacity
    self.slots = [ 100000000 for _ in range(capacity) ]
    self.pointer = 0

  """
  return: should_top, best_index, best_value
  """
  def should_stop(self, new_value):
    # if capacity is 0
    # no early stop
    if self.capacity == 0:
      return False, None, None

    # set new value
    curr_idx = self.pointer % self.capacity
    self.pointer += 1
    self.slots[ curr_idx ] = new_value

    pivot_idx = (curr_idx + 1) % self.capacity # right side
    pivot_value = self.slots[ pivot_idx ]
    # check if all values never imporved
    # in terms of pivot value
    has_drop = False
    for idx, value in enumerate(self.slots):
      if idx != pivot_idx:
        if pivot_value > value:
          has_drop = True
          break

    if has_drop:
      # it dropped, so just keep training
      return False, None, None
    else:
      # no drop, pivot is the best
      return True, max(0, self.pointer - self.capacity), pivot_value



def mkdir(d):
  if not os.path.exists(d):
    os.makedirs(d)


def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  print(msg)



def get_model(FLAGS, w2v_weights):
  ### get model
  model = CNN(dtype=tf.float32,
                filter_widths=FLAGS.filter_width,
                num_filters=FLAGS.num_filters,
                l2_reg=FLAGS.l2_reg,
                keep_prob=FLAGS.keep_prob,
                w2v_weights=w2v_weights,
                tune_embedding=FLAGS.tune_embedding,
                do_max_pool=FLAGS.do_max_pool)

  return model


# def validation(model, sess, dataset):
#   num_data = dataset.num_data
#   total_loss = 0.0

#   labels = np.zeros((num_data))
#   preds = np.zeros((num_data))
#   scores = np.zeros((num_data))

#   for idx, (data, label) in enumerate(dataset.iter()):
#     loss, score, pred = model.inference_with_labels(sess, data, label)

#     total_loss += np.mean(loss)

#     labels[idx] = label[0]
#     scores[idx] = np.mean(score)
#     preds[idx] = np.round(scores[idx])
    

#   accuracy = accuracy_score(labels, preds)

#   return total_loss/num_data, accuracy


def validation(model, sess, dataset):
  dataset.reset()
  num_data = dataset.num_data
  total_loss = 0.0

  labels = np.zeros((num_data))
  preds = np.zeros((num_data))
  scores = np.zeros((num_data))

  # for idx, (data, label, lengths, cnt_epoch) in enumerate(dataset.iter()):
  #   loss, score, pred = model.inference_with_labels(sess, data, label, lengths)

  #   total_loss += np.mean(loss)

  #   labels[idx] = label[0]
  #   scores[idx] = np.mean(score)
  #   preds[idx] = np.round(scores[idx])

  sidx = 0
  eidx = 0

  for data, label, lengths, cnt_epoch in dataset.iter_batch():
    loss, score, pred = model.inference_with_labels(sess, data, label, lengths)
    samples = data.shape[0]

    total_loss += loss*samples

    sidx = eidx
    eidx += samples
    labels[sidx:eidx] = label
    preds[sidx:eidx] = pred
    scores[sidx:eidx] = score
    

  accuracy = accuracy_score(labels, preds)

  return total_loss/num_data, accuracy




def main(sys_argv):
  FLAGS, rest_args = arg_parse(sys_argv)
  
  ### prepare directories
  mkdir(FLAGS.train_dir)
  FLAGS.checkpoint_path = os.path.join(FLAGS.train_dir, 'ckpt', 'model')
  mkdir(os.path.join(FLAGS.train_dir, 'ckpt'))
  FLAGS.log_dir = os.path.join(FLAGS.train_dir)
  FLAGS.summary_dir = os.path.join(FLAGS.train_dir, 'summary')
  mkdir(FLAGS.summary_dir)

  logging("[%s: INFO] Setup: %s" % 
              (datetime.now(), str(FLAGS)), FLAGS)


  ### get dataset
  ### get w2v
  w2v_weights = np.load( os.path.join(FLAGS.w2v_dir, "weights.npy") )
  with open(os.path.join(FLAGS.w2v_dir, "dict.pkl"), 'r') as fr:
    w2v_dict = pickle.load(fr)

  ### get dataset
  train_set = Dataset(FLAGS.train_file, w2v_weights, w2v_dict, 
                      batch_size=FLAGS.batch_size,
                      max_epoch=FLAGS.max_epoch,
                      need_shuffle=True)

  valid_set = Dataset(FLAGS.val_file, w2v_weights, w2v_dict, 
                      batch_size=FLAGS.batch_size,
                      max_epoch=1,
                      need_shuffle=False)
  
  # valid_set = ValidDataset(FLAGS.val_file, w2v_weights, w2v_dict)

  num_features = train_set.get_feature_shape()[1]


  with tf.Graph().as_default():
    model = get_model(FLAGS, w2v_weights)
    monitor = ValidationMonitor(capacity=FLAGS.validation_capacity)
    monitor = ValidationMonitor(capacity=0)


    with tf.Session() as sess:
      learning_rate = FLAGS.learning_rate

      saver = tf.train.Saver(max_to_keep=FLAGS.validation_capacity)
      writer = tf.summary.FileWriter(FLAGS.summary_dir)

      sess.run( tf.global_variables_initializer() )

      for step, (data, labels, lengths, cnt_epoch) in enumerate(train_set.iter_batch()):
        start_time = time.time()

        loss, scores, hits, summary = model.train(sess, data, labels, lengths, learning_rate)

        duration = time.time() - start_time


        if step % FLAGS.log_interval == 0:
          examples_per_sec = FLAGS.batch_size / float(duration)

          logging("[%s: INFO] %d step => loss: %.3f, acc: %.3f (%.1f examples/sec; %.3f sec/batch)" % 
            (datetime.now(), step, loss, hits/FLAGS.batch_size, examples_per_sec, duration), FLAGS)

          writer.add_summary(summary, step)
          # print('pred', pred)
          # print('labels', labels)

        ### validation
        if cnt_epoch is not None:
          logging("[%s: INFO] %d epoch done!" % 
              (datetime.now(), cnt_epoch), FLAGS)

          saver.save(sess, FLAGS.checkpoint_path, global_step=cnt_epoch)
          
          loss, accuracy = validation(model, sess, valid_set)

          logging("[%s: INFO] Validation Result at %d epochs: loss: %.3f, accuracy: %.3f" % 
            (datetime.now(), cnt_epoch, loss, accuracy), FLAGS)

          valid_summary = model.summary_valid_loss(sess, loss)
          writer.add_summary(valid_summary, step)

          ### early stop checking
          should_stop, best_idx, best_loss = monitor.should_stop(loss)

          if should_stop:
            logging("[%s: INFO] Early Stopping at %d. Best was %d with loss %.3f" % 
              (datetime.now(), cnt_epoch, best_idx, best_loss), FLAGS)
            break







if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)
