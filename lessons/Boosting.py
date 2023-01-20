import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

  dt_heart = pd.read_csv('./data/heart.csv')

  X = dt_heart.drop(['target'], axis=1)
  y = dt_heart['target']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  #Use tree within it
  boosting = GradientBoostingClassifier(n_estimators = 50).fit(X_train, y_train)
  boosting_predictions = boosting.predict(X_test)
  print("="*64)
  print(accuracy_score(boosting_predictions, y_test))