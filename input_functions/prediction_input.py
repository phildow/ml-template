import tensorflow as tf
import numpy as np

# training input dataset and parsing

def _parser(protobuf, params):
  features = {
    # 'x': tf.FixedLenFeature([], tf.string, default_value=""),
  }
  
  parsed_features = tf.parse_single_example(protobuf, features)
  x = parsed_features["x"]
  
  return {'x':x}

def input_fn(filenames, batch_size=32, params={}):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x:_parser(x, params))
  dataset = dataset.batch(batch_size)

  return dataset
