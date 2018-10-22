import tensorflow as tf
import numpy as np

# uncomment if you are using keras
# from tensorflow.python.keras import backend as K

def model_fn(features, labels, mode, params):
  
  # uncomment if you are using keras
  # if mode == tf.estimator.ModeKeys.TRAIN:
  #   K.set_learning_phase(True)
  # else:
  #   K.set_learning_phase(False)

  # build model layers

  # layers = tf.reshape(...)
  # layers = tf.keras.layers.x(...)(layers)
  # outputs = tf.keras.layers.x(...)layers)

  # prepare predictions

  predictions = {
    
  }
  prediction_output = tf.estimator.export.PredictOutput({
    
  })

  # return an estimator spec for prediction before computing a loss

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode, 
      predictions=predictions,
      export_outputs={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output
      })

  # calculate loss

  # loss = tf.losses.softmax_cross_entropy(...)

  # calculate accuracy metric

  # accruacy = tf.metrics.accuracy(...)

  if mode == tf.estimator.ModeKeys.TRAIN:

    # generate some summary info

    # tf.summary.scalar('average_loss', loss)
    # tf.summary.scalar('accuracy', accuracy[1])

    # prepare an optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())

    # return an estimator spec

    return tf.estimator.EstimatorSpec(
      mode=mode, 
      loss=loss, 
      train_op=train_op)
  
  if mode == tf.estimator.ModeKeys.EVAL:

    # add evaluation metrics
    
    eval_metric_ops = {
      "accuracy": accuracy
    }

    # return an estimator spec

    return tf.estimator.EstimatorSpec(
      mode=mode, 
      loss=loss, 
      eval_metric_ops=eval_metric_ops)