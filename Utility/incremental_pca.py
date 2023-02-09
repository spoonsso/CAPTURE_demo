from sklearn.decomposition import IncrementalPCA

def incremental_pca(input):
    n_components = 10
    batch_size = 100
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_pca = ipca.fit_transform(input)
    return X_pca