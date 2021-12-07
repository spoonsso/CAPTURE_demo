from sklearn.decomposition import IncrementalPCA
import numpy as np

def incremental_pca(input, n_components):
    batch_size = 20
    # np.shape(input)


    print("incremental_pca_running")
    # np_input_test = np.array(input._data).reshape(input.size[::-1]).T
    # print(np_input_test)
    # np_input = np.asarray(input)
    # np_input = np_input.reshape(input.size).transpose()
    # input_t = np.reshape(np.asarray(input),(size1,size2))

    ipca = IncrementalPCA(n_components=n_components)#, batch_size=batch_size)
    X_pca = ipca.fit_transform(input)
    explained = ipca.explained_variance_ratio_
    coeffs = ipca.components_
    return [X_pca, coeffs, explained]