import pandas as pd
import matplotlib.pyplot as plt
from constants import MODEL_NAME
import tensorflow.keras as keras

from os.path import exists
from typing import Optional

class Path:
  def plots(self, path: Optional[str]):
    return self.storage(f'plots/{path}')

  def board(self, path: Optional[str]):    
    return self.storage(f'board/{path}')

  def logs(self, path: Optional[str]):    
    return self.storage(f'logs/{path}')

  def storage(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'storage{path}'

  def resources(self, path: Optional[str]):    
    path = self.clean_path(path) 

    return f'resources{path}'

  def clean_path(self, path: Optional[str]):
    if path is None:
      return ''

    if path.endswith('/') is True:
      path = path[:-1]

    if path.startswith('/') is True:
      return path 

    return f'/{path}'

path = Path()

def load_model():
  path = f'storage/{MODEL_NAME}'
  model_exists = exists(path)

  if (model_exists):
    return keras.models.load_model(path)

  model = keras.models.Sequential()

  model.add(keras.layers.Dense(64, input_dim=4, activation='relu'))
  model.add(keras.layers.Dense(32, activation='relu'))
  model.add(keras.layers.Dense(6))

  model.compile(optimizer='adam', loss='mse')

  return model

def get_df(): 
  dataset = pd.read_csv(path.resources('clean_dataset.csv'))

  all_numbers = pd.concat([dataset[col] for col in dataset.columns])

  number_counts = all_numbers.value_counts().sort_index()
  number_counts = number_counts.reindex(range(1, 60), fill_value=0)

  plt.figure(figsize=(20, 6))
  number_counts.plot(kind='bar', color='skyblue')
  plt.title('Frequency of Numbers in num1 to num6')
  plt.xlabel('Number')
  plt.ylabel('Frequency')
  plt.xticks(rotation=90)
  plt.grid(axis='y')
  plt.savefig(path.plots('dataset/numbers-frequency.png'))

  return dataset
