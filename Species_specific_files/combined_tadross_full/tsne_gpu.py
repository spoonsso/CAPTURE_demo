import tsnecuda as tc

def tsne_gpu(x_train):
    tsne = tc.TSNE(n_iter=5000, verbose=2, num_neighbors=100)
    tsne_results = tsne.fit_transform(x_train)
    return tsne_results
