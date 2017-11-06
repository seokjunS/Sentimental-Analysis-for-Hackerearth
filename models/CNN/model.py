import tensorflow as tf




"""
CNN model for sentimental analysis
"""
class Model(object):
  def __init__(self,
                dtype,
                filter_widths,
                len_seq,
                num_filters,
                num_labels,
                l2_reg,
                keep_prob,
                w2v_weights,
                tune_embedding=False):
    self.inputs = tf.placeholder(tf.int32, shape=[None, len_seq], name='feature_placeholder')
    self.labels = tf.placeholder(tf.int32, shape=[None], name='label_placeholder')
    self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training_placeholder')
    self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr_placeholder')

    self.filter_widths = filter_widths
    self.num_filters = num_filters
    self.num_labels = num_labels
    self.len_seq = len_seq
    self.l2_reg = l2_reg
    self.keep_prob = keep_prob
    self.w2v_weights = w2v_weights
    self.tune_embedding = tune_embedding

    self._build()


  def _build(self):
    inputs = self.inputs

    ### embedding
    # with tf.device("/cpu:0"):
    embeddings = tf.get_variable("embedding",
      shape=self.w2v_weights.shape,
      dtype=tf.float32,
      initializer=tf.constant_initializer( self.w2v_weights ),
      trainable=self.tune_embedding )
    
    # (batch, len, emb)
    inputs = tf.nn.embedding_lookup(embeddings, inputs)
    inputs = tf.expand_dims(inputs, axis=[1])

    ### convolution
    x = tf.identity( inputs )

    # convolution
    self.conv_acts = []
    for i, filter_width in enumerate(self.filter_widths):
      num_filter = self.num_filters[i]
      name = "conv_%d" % i
      with tf.variable_scope(name):
        # define conv
        weights = tf.get_variable("weights", 
                                  shape=[1, filter_width, x.get_shape().as_list()[-1], num_filter],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  # regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg)
                                  )
        # Create variable named "biases".
        biases = tf.get_variable("biases", 
                                  shape=[num_filter],
                                  dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(inputs, 
                            filter=weights,
                            strides=[1, 1, 1, 1], 
                            padding='SAME')
        act = conv + biases

        act = tf.layers.batch_normalization(inputs=act, 
                                            training=self.is_training)
        act = tf.nn.relu(act)

        # (batch, 1, 1, # filters)
        act = tf.nn.max_pool(act,
                                  ksize=[1, 1, self.len_seq, 1],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID')
        # (batch, #filters)
        act = tf.squeeze( act, axis=[1, 2] )

        # act = tf.reduce_max(act, axis=2)
        # act = tf.squeeze( act, axis=[1] )
        

        self.conv_acts.append( act )


    with tf.variable_scope("concat"):
      x = tf.concat( self.conv_acts, axis=-1 )
      # tf.summary.histogram('concat_act', x)
      x = tf.layers.dropout(x, 
                            rate=self.keep_prob, 
                            training=self.is_training)



    with tf.variable_scope("softmax"):
      # x: (batch, # filters)
      weights = tf.get_variable("weights",
                                shape=[x.get_shape().as_list()[-1], self.num_labels],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))

      biases = tf.get_variable("biases",
                                shape=[self.num_labels],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

      # (batch, # labels)
      self.logits = tf.matmul(x, weights) + biases

    # loss
    with tf.variable_scope("etc"):
      softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
      # tf.summary.scalar('cross_loss', softmax_loss)

      reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      # tf.summary.scalar('reg_loss', reg_loss)

      self.loss = softmax_loss + reg_loss
      # tf.summary.scalar('total_loss', self.loss)

      # optimizer = tf.train.AdamOptimizer( self.learning_rate )
      # optimizer = tf.train.RMSPropOptimizer( self.learning_rate )
      optimizer = tf.train.AdadeltaOptimizer( self.learning_rate )
      grd = optimizer.compute_gradients( self.loss )
      self.train_op = optimizer.apply_gradients(grd)

      self.pred_op = tf.cast( tf.argmax(self.logits, axis=-1), tf.int32 )

      self.hit_op = tf.reduce_sum( tf.cast( tf.equal( self.pred_op, self.labels ) , tf.float32) )

      # self.summary_op = tf.summary.merge_all()


  def train(self, sess, d_features, d_labels, learning_rate):
    # _, loss_val, hit_val, pred, summary = sess.run([self.train_op, self.loss, self.hit_op, self.pred_op, self.summary_op], feed_dict={
    #   self.inputs: d_features,
    #   self.labels: d_labels,
    #   self.is_training: True,
    #   self.learning_rate: learning_rate
    # })

    # return loss_val, hit_val, pred, summary
    _, loss_val, hit_val, pred = sess.run([self.train_op, self.loss, self.hit_op, self.pred_op], feed_dict={
      self.inputs: d_features,
      self.labels: d_labels,
      self.is_training: True,
      self.learning_rate: learning_rate
    })

    return loss_val, hit_val, pred


  def inference_with_hits(self, sess, d_features, d_labels):
    pred, hit_val = sess.run([self.pred_op, self.hit_op], feed_dict={
      self.inputs: d_features,
      self.labels: d_labels,
      self.is_training: False
    })

    return pred, hit_val


  def inference(self, sess, d_features):
    pred = sess.run([self.pred_op], feed_dict={
      self.inputs: d_features,
      self.is_training: False
    })

    return pred
