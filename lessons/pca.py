import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#To identify what is the maing script of this file.
#If this is the main script, follow the next process
if __name__ == "__main__":
  dt_heart = pd.read_csv("./data/heart.csv")

  print(dt_heart.head(5))

  #Save dataset without target column
  dt_features = dt_heart.drop(["target"], axis=1)

  #Save target column
  dt_target = dt_heart['target']

  #Scale features
  dt_features = StandardScaler().fit_transform(dt_features)

  #Split dataset
  X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

  print(X_train.shape)
  print(y_train.shape)

  #n_components = min(n_muestras, n_features)
  pca = PCA(n_components = 3)
  pca.fit(X_train)

  # It does not send all data for training at the same time
  # It send small blocks of data for training and combining for the final result
  ipca = IncrementalPCA(n_components = 3, batch_size = 10)
  ipca.fit(X_train)
  
  #See variance
  #X-axis represent the number of components (0,1,2)
  #Y-axis how important of each component
  #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)

  #For this case, 1 y 2 components are the most important for our model
  #plt.show()

  #Crate the clasifier
  logistic = LogisticRegression(solver='lbfgs')

  dt_train = pca.transform(X_train)
  dt_test = pca.transform(X_test)
  logistic.fit(dt_train, y_train)
  print("SCORE PCA: ", logistic.score(dt_test, y_test))

  dt_train = ipca.transform(X_train)
  dt_test = ipca.transform(X_test)
  logistic.fit(dt_train, y_train)
  print("SCORE IPCA: ", logistic.score(dt_test, y_test))

  #To graph the accuracy between the two modes of PCA
  pca_data = {'accuracy':[], 'components': []}
  ipca_data = {'accuracy':[], 'components': []}

  for i in range(2, 10):
    pca = PCA(n_components = i)
    pca.fit(X_train)

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    accuracy = logistic.score(dt_test, y_test)

    pca_data['accuracy'].append(accuracy)
    pca_data['components'].append(i)

#IPCA
  for i in range(2, 10):
    ipca = IncrementalPCA(n_components = i, batch_size = 10)
    ipca.fit(X_train)

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    accuracy = logistic.score(dt_test, y_test)

    ipca_data['accuracy'].append(accuracy)
    ipca_data['components'].append(i)

  #Now, let's plot the values
  plt.plot(pca_data['components'], pca_data['accuracy'], label="PCA")
  plt.plot(ipca_data['components'], ipca_data['accuracy'], label="IPCA")

  plt.title("PCA vs IPCA")
  plt.xlabel = "Number of components"
  plt.ylabel = "Accuracy"
  plt.legend()
  plt.show()