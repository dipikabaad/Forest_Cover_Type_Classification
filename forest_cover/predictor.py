import os
import pickle
import pandas as pd
import numpy as np

import tensorflow as tf

class ForestPredictor(object):
  def __init__(self, model, preprocessor):
    self._model = model
    self._preprocessor = preprocessor

  def predict(self, instances, **kwargs):
    data = pd.read_json(instances, orient='index')
    ids = data['Id']
    preprocessed_data = self._preprocessor.preprocess(data)
    
    ds = tf.data.Dataset.from_tensor_slices((dict(preprocessed_data)))
    batch_size = 30
    ds = ds.batch(batch_size)
    outputs = np.argmax(self._model.predict(ds), axis=-1)
    final_df = pd.DataFrame()
    final_df = final_df.from_dict({'Id': ids, 'Cover_Type': outputs})
    json_df = final_df.to_json(orient='index')
    return json_df

  @classmethod
  def from_path(cls, model_dir):
    model_path = os.path.join(model_dir, 'forest_model_layer_100_50_epoch_100.h5')
    model = tf.keras.models.load_model(model_path)

    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
      preprocessor = pickle.load(f)

    return cls(model, preprocessor)