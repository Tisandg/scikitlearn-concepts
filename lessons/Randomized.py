import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
  
  dataset = pd.read_csv("./data/felicidad.csv")

  print(dataset)

  X = dataset.drop(['score','rank', 'country'], axis=1)
  y = dataset['score']

  reg = RandomForestRegressor()

  parametros={
    'n_estimators': range(4, 16),
    'criterion': ['mse','mae'],
    'max_depth': range(2, 11),
  }

  rand_est = RandomizedSearchCV(reg, parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)

  print("Best estimator")
  print(rand_est.best_estimator_)
  print("Best params")
  print(rand_est.best_params_)
  print(rand_est.predict(X.loc[[0]]))
  print(y.loc[0])