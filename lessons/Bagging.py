import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

  dt_heart = pd.read_csv('./data/heart.csv')

  X = dt_heart.drop(['target'], axis=1)
  y = dt_heart['target']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  knn_class = KNeighborsClassifier().fit(X_train, y_train)
  knn_pred = knn_class.predict(X_test)
  print("="*30)
  print(accuracy_score(knn_pred, y_test))

  bag_class = BaggingClassifier(base_estimator = KNeighborsClassifier(), n_estimators = 50).fit(X_train, y_train)
  bag_predictions = bag_class.predict(X_test)
  print("="*30)
  print(accuracy_score(bag_predictions, y_test))

  # Now, test with more estimators
  classifiers = {
    "DecisionTree": DecisionTreeClassifier(),
    "SVC": SVC(),
    "SGDClassifier": SGDClassifier()
  }

  for name, clf in classifiers.items():
    bag_class = BaggingClassifier(base_estimator = clf, n_estimators = 50).fit(X_train, y_train)
    bag_predictions = bag_class.predict(X_test)
    print("="*30)
    print("Model ", name, " obtained ", accuracy_score(bag_predictions, y_test), "of accuracy")