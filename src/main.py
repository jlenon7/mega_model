import time
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import MODEL_NAME
from helpers import path, get_df, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

dataset = get_df()

X = dataset.drop(columns=['contest', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6'])
y = dataset[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = load_model()

model.fit(
  x=X_train,
  y=y_train,
  validation_data=(X_test,y_test),
  batch_size=10,
  epochs=1000,
  callbacks=[
    EarlyStopping(
      monitor='val_loss',
      mode='min',
      patience=40
    ),
    TensorBoard(
      log_dir=path.board(f'fit-{calendar.timegm(time.gmtime())}'),
      histogram_freq=1,
      write_graph=True,
      write_images=True,
      update_freq='epoch',
      profile_batch=2,
      embeddings_freq=1
    )
  ]
)

metrics = pd.DataFrame(model.history.history)

metrics \
  .plot() \
  .figure \
  .savefig(path.plots('model/is-overfitting-train-test-data.png'))

predictions = model.predict(X_test)

print()
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, predictions))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, predictions))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, predictions)))
print('Explained Variance Regression Score:', explained_variance_score(y_test, predictions))
print('R-squared (R²):', r2_score(y_test, predictions))

plt.figure(figsize=(20, 6))
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs = axs.flatten()
titles = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']

for i in range(6):
    axs[i].scatter(y_test.iloc[:, i], predictions[:, i], alpha=0.5)
    axs[i].set_xlabel('Actual Values')
    axs[i].set_ylabel('Predicted Values')
    axs[i].set_title(f'Predicted vs Actual for {titles[i]}')
    axs[i].plot([0, 100], [0, 100], 'r--')

plt.tight_layout()
plt.savefig(path.plots('model/predictions.png'))

print()
print('Saving the model at', path.storage(MODEL_NAME))
model.save(path.storage(MODEL_NAME))
