import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.metrics

from autosklearn.classification import AutoSklearnClassifier

if __name__ == "__main__":
  
  dataset = pd.read_csv('./data/heart.csv')

  X = dataset.drop(['target'], axis=1)
  y = dataset['target']

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
  automl = AutoSklearnClassifier()
  automl.fit(X_train, y_train)
  y_hat = automl.predict(X_test)
  print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))