import random
import numpy as np
from helpers import get_df, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset = get_df()

X = dataset.drop(columns=['contest', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6'])
y = dataset[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = load_model()

predictions = model.predict(X_test, verbose=0)

print()
for i in range(10):
  length = len(predictions)
  index = random.randint(0, length)
  print(f'Predicted: {np.round(predictions[index])}, Actual: {y_test.iloc[index].values}')
