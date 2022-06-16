import numpy as np
import faiss

class KNNEmbed:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X,dtype=np.float32))
        self.y = np.ascontiguousarray(y,dtype=np.float32)

    def predict(self, X, weights='standard'):
        print("Predicting")
        distances, indices = self.index.search(np.ascontiguousarray(X,dtype=np.float32), k=self.k)
        votes = self.y[indices]

        if weights=='distance':
            min_dist = np.min(distances[np.nonzero(distances)])/2
            distances = np.clip(distances, min_dist, None)
            weights = 1/distances
            weights = weights/np.repeat(np.expand_dims(np.sum(weights, axis=1), axis=1), self.k, axis=1)
        else:
            weights = 1/self.k

        weights = np.repeat(np.expand_dims(weights, axis=2), 2, axis=2)
        predictions = np.sum(votes*weights, axis=1)
        return predictions