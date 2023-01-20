import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
  dataset = pd.read_csv('./data/candy.csv')

  X = dataset.drop(['competitorname'], axis=1)

  kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
  print("Total centros:", len(kmeans.cluster_centers_))
  print("="*64)
  print(kmeans.predict(X))

  dataset['group'] = kmeans.predict(X)

  
  sns.scatterplot(data=dataset, x="sugarpercent", y="winpercent", hue="group",palette="deep")
  plt.show()