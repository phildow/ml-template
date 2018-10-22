import tensorflow as tf
import numpy as np

# training input dataset and parsing

def _parser(protobuf, params):
  features = {
    # 'x': tf.FixedLenFeature([], tf.string, default_value=""),
    # 'y': tf.FixedLenFeature([], tf.int64, default_value=0)
  }
  
  parsed_features = tf.parse_single_example(protobuf, features)

  x = parsed_features["x"]
  y = parsed_features["y"]
  
  return {'x':x}, y

def input_fn(filenames, batch_size=32, num_epochs=1000, params={})):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(lambda x:_parser(x, params))
  dataset = dataset.shuffle(1024)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)
  
  return dataset
