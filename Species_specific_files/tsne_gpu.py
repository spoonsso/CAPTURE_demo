import tsnecuda as tc

def tsne_gpu(x_train):
    tsne = tc.TSNE(n_iter=1000, verbose=2, num_neighbors=64)
    tsne_results = tsne.fit_transform(x_train)
    return tsne_results
