import sys
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse
import time
import random
import pickle

sys.path.append( os.path.join(os.path.dirname(__file__), "..") )
from utils.data_preprocess import Dataset, mkdir
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
      default=os.path.join(DATA_PATH, 'WV-500000'),
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
      default=0.5,
      help='Rate to be kept for dropout.'
  )
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_epoch',
      type=int,
      default=30,
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
      '--prev_checkpoint_path',
      type=str,
      default='',
      help='Restore from pre-trained model if specified.'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default=os.path.join(TRAIN_PATH, 'test', 'model'),
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(TRAIN_PATH, 'test'),
      help='Directory for log data.'
  )
  parser.add_argument(
      '--summary_dir',
      type=str,
      default=os.path.join(TRAIN_PATH, 'test', 'summary'),
      help='Directory for log data.'
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
      default=[1, 3, 5],
      help='Space seperated list of filter widths. (ex, --filter_width 4 8 12)'
  )
  parser.add_argument(
      '--num_filters',
      type=int,
      nargs='+',
      default=[200, 200, 200],
      help='Space seperated list of number of filters. (ex, --num_filters 4 8 12)'
  )
  parser.add_argument(
      '--l2_reg',
      type=float,
      default=0.01,
      help=''
  )

  FLAGS, unparsed = parser.parse_known_args(args)

  return FLAGS, unparsed





def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  print(msg)



def inference_with_hits(dataset, model, sess):
  dataset.init(sess)
  total_preds = []
  total_hits = 0
  total_count = 0.0
  while True:
    try:
      labels, features = sess.run(dataset.get_next())

      pred, hits = model.inference_with_hits(sess, features, labels)

      total_hits += hits
      total_preds.append( pred )
      total_count += features.shape[0]
    except tf.errors.OutOfRangeError: # one epoch finish
      break

  return total_preds, total_hits, total_count




def main(sys_argv):
  FLAGS, rest_args = arg_parse(sys_argv)
  
  ### prepare directories
  mkdir(FLAGS.log_dir)
  logging("[%s: INFO] Setup: %s" % 
              (datetime.now(), str(FLAGS)), FLAGS)


  with tf.Graph().as_default():
    ### get w2v
    w2v_weights = np.load( os.path.join(FLAGS.w2v_dir, "weights.npy") )
    with open(os.path.join(FLAGS.w2v_dir, "dict.pkl"), 'r') as fr:
      w2v_dict = pickle.load(fr)

    ### get dataset
    train_set = Dataset(FLAGS.train_file, w2v_weights, w2v_dict, batch_size=FLAGS.batch_size)
    valid_set = Dataset(FLAGS.val_file, w2v_weights, w2v_dict, batch_size=FLAGS.batch_size)

    _, data_length = train_set.get_feature_shape()


    ### get model
    model = Model(dtype=tf.float32,
                  filter_widths=FLAGS.filter_width,
                  len_seq=data_length,
                  num_filters=FLAGS.num_filters,
                  num_labels=2,
                  l2_reg=FLAGS.l2_reg,
                  keep_prob=FLAGS.keep_prob,
                  w2v_weights=w2v_weights,
                  tune_embedding=FLAGS.tune_embedding)

    with tf.Session() as sess:
      step = 0
      prev_acc = 0.0
      max_acc = 0.0
      learning_rate = FLAGS.learning_rate

      saver = tf.train.Saver()
      # writer = tf.summary.FileWriter(FLAGS.summary_dir)

      sess.run( tf.global_variables_initializer() )

      for cnt_epoch in range(FLAGS.max_epoch):
        train_set.init(sess, seed=random.randint(0, 10000000))

        while True:
          try:
            labels, features = sess.run(train_set.get_next())

            start_time = time.time()

            loss, hits, pred = model.train(sess, features, labels, learning_rate)
            
            duration = time.time() - start_time
            
            step += 1

            if step % FLAGS.log_interval == 0:
              examples_per_sec = FLAGS.batch_size / float(duration)

              logging("[%s: INFO] %d step => loss: %.3f, acc: %.3f (%.1f examples/sec; %.3f sec/batch)" % 
                (datetime.now(), step, loss, hits/FLAGS.batch_size, examples_per_sec, duration), FLAGS)

              # writer.add_summary(summary, step)
              # print('pred', pred)
              # print('labels', labels)


          except tf.errors.OutOfRangeError: # one epoch finish
            logging("[%s: INFO] %d epoch done!" % 
                (datetime.now(), cnt_epoch), FLAGS)

            saver.save(sess, FLAGS.checkpoint_path, global_step=cnt_epoch)
            break

        ### validation
        if cnt_epoch % FLAGS.validation_interval == 0 or cnt_epoch == FLAGS.max_epoch-1:
          _, num_hits, counts = inference_with_hits(valid_set, model, sess)
          curr_acc = num_hits / counts

          logging("[%s: INFO] Validation Result at %d epochs: %d among %d sentences: %.3f" % 
            (datetime.now(), cnt_epoch, num_hits, counts, curr_acc), FLAGS)

          # if np.absolute(curr_acc - prev_acc) < 0.001: # diff < 0.1%
          #   learning_rate /= 2
          #   logging("[%s: INFO] Learning rate decaying! => %.5f" % 
          #     (datetime.now(), learning_rate), FLAGS)

          # if max_acc - curr_acc > 0.05 or learning_rate < 0.0001:  
          #   # significant drop over 5% or too small lr
          #   logging("[%s: INFO] Stop! max: %.3f, prev: %.3f, current: %.3f" % 
          #     (datetime.now(), max_acc, prev_acc, curr_acc), FLAGS)
          #   break

          # max_acc = max(max_acc, curr_acc)
          # prev_acc = curr_acc 







if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)
