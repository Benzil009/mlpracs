import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = np.array([
    [1,2],[2,2],[2,3],[8,7],[8,8],[25,80],
    [7,7],[8,6],[1,1],[9,8]
])

model = DBSCAN(eps=3, min_samples=2)
labels = model.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='plasma')
plt.title("DBSCAN Clustering (10 points)")
plt.show()
