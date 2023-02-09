import tsnecuda as tc
import h5py
import numpy as np
from tqdm import tqdm

# def read_mat(file_path):
#     f = h5py.File(file_path)
#     arrays = {}
#     for key, values in tqdm(f.items()):
#         print(key)
#         arrays[key] = np.array(values)
#     return arrays

analysis_file_path = './analysisstruct_notsne_2.mat'
predictions_file_path = '/hpc/group/tdunn/st3dio/analysis/PDb/merged_predictions.mat'
analysisstruct = load_mat(analysis_file_path)
# predictions = read_mat(predictions_file_path)

# f = h5py.File(analysis_file_path)
# print(f.items())
# jt_features = np.array(f['jt_features'])
# print(np.shape(jt_features))



# num_neighbors = [50,100]
# perplexity = [70,400000]

# for nn in num_neighbors:
#     for p in perplexity:
#         try:
#             tsne = tc.TSNE(n_iter=1000, verbose=2, num_neighbors=nn,perplexity=p)
#             tsne_results = tsne.fit_transform(jt_features.T)
#             print("worked on " + str(nn) + " " + str(p))
#         except:
#             print("failed on " + str(nn) + " " + str(p))
#             continue

## Clustering
#pip install mpl-scatter-density
# Could also just do np.2dhistogram with high bins and interpolate for watershed clustering