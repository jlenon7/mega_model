import pandas as pd
from helpers import path

data = pd.read_csv(path.resources('dataset.csv'))

numbers_df = data['numbers'].str.split(',', expand=True)
numbers_df.columns = [f'num{i+1}' for i in range(numbers_df.shape[1])]

numbers_df = numbers_df.apply(pd.to_numeric)
transformed_data = pd.concat([data['contest'], numbers_df], axis=1)

transformed_data['sum'] = numbers_df.sum(axis=1)
transformed_data['mean'] = numbers_df.mean(axis=1)
transformed_data['std'] = numbers_df.std(axis=1)

transformed_data.to_csv(path.resources('clean_dataset.csv'))
