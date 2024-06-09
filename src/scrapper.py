import requests
import pandas as pd
from helpers import path
import os.path as os_path

contest = 1
has_error = False

numbers = []
contests = []

if os_path.exists(path.resources('dataset.csv')):
  df = pd.read_csv(path.resources('dataset.csv'))
  numbers = df['numbers'].values.tolist()
  contests = df['contest'].values.tolist()
  contest = df['contest'].iloc[-1] + 1

while has_error == False:
  print(f'searching for contest number {contest}')

  response = requests.get(f'https://servicebus2.caixa.gov.br/portaldeloterias/api/megasena/{contest}')
 
  if response.status_code != 200:
    pd.DataFrame({ 'contest': contests, 'numbers': numbers }).to_csv(path.resources('dataset.csv'))
    print()
    print(f'request failed with status code {response.status_code}')
    print(f'response content: {response.content}')
    continue

  number = response.json().get('listaDezenas')

  print(f'found {number} numbers in contest {contest}')

  contests.append(contest)
  numbers.append(','.join(map(str, number)))

  pd.DataFrame({ 'contest': contests, 'numbers': numbers }).to_csv(path.resources('dataset.csv'))

  contest += 1
