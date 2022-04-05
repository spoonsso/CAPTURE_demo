import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
import numpy as np

def embed_scatter(data, 
                  filename = './embedding_scatter',
                  save = True):
    f = plt.figure()
    plt.scatter(data[:,0], data[:,1], marker='.', s=3, linewidths=0)
    plt.title = filename
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if save:
        plt.savefig(''.join([filename,'.png']))

def clustering(data, filename):
    x_range = int(np.ceil(np.amax(data[:,0])) - np.floor(np.amin(data[:,0])))
    y_range = int(np.ceil(np.amax(data[:,1])) - np.floor(np.amin(data[:,1])))
    hist,xedges,yedges = np.histogram2d(data[:,0], data[:,1], bins=[1000, 1000],
                          range=[[int(np.floor(np.amin(data[:,0]))-x_range/20),int(np.ceil(np.amax(data[:,0]))+x_range/20)],
                                 [int(np.floor(np.amin(data[:,1]))-y_range/20),int(np.ceil(np.amax(data[:,1]))+y_range/20)]],
                          density=True)
    hist = np.rot90(hist)

    assert xedges[0]<xedges[-1] and yedges[0]<yedges[1]

    x_bin_idx = np.zeros(np.shape(data[:,0]))
    y_bin_idx = np.zeros(np.shape(data[:,1]))
    for i in range(len(x_bin_idx)):
        x_bin_idx[i] = 1001-np.argmax(xedges>data[i,0])
        y_bin_idx[i] = 1001-np.argmax(yedges>data[i,1])

    gauss_filt_hist = gaussian_filter(hist, sigma=20)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(gauss_filt_hist)
    ax.set_aspect('auto')
    plt.savefig(''.join([filename,'_2dhist_gauss.png']))
    plt.close()

    watershed_map = watershed(-gauss_filt_hist,connectivity=8, watershed_line=True)
    watershed_borders = np.where(watershed_map==0,1,0)
    print(watershed_borders)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(watershed_borders, cmap='gray_r')
    ax.set_aspect('auto')
    plt.savefig(''.join([filename,'_watershed.png']))
    plt.close()
    data_by_cluster = watershed_map[x_bin_idx.astype(int),y_bin_idx.astype(int)]
    print(str(int(np.amax(data_by_cluster))),"clusters detected")

    return watershed_map, data_by_cluster

def sample_clusters(data, data_by_cluster, size=30):
    data = np.append(data,np.expand_dims(np.arange(np.shape(data)[0]),axis=1),axis=1)
    sampled_points = np.empty((0,np.shape(data)[1]))
    for cluster_id in np.unique(data_by_cluster):
        points = data[data_by_cluster==cluster_id,:]
        if len(points)==0:
            continue
        else:
            num_points = min(len(points),size)
            sampled_points = np.append(sampled_points, 
                                       np.random.permutation(points)[:num_points], 
                                       axis=0)
    print(sampled_points.shape)
    return sampled_points[:,:-1],np.squeeze(sampled_points[:,-1]).astype(int).tolist()

def reembed(template, template_idx, full_data, method='tsne_cuda', plot_folder='./plots/'):
    import time
    print("Finding template embedding using ", method)
    if method == 'tsne_cuda':
        n = np.shape(template)[0]
        import tsnecuda as tc
        tsne = tc.TSNE(n_iter=1000, verbose=2, num_neighbors=100, perplexity=int(n/100), learning_rate=int(n/12))
        temp_embedding = tsne.fit_transform(template)
        filename = ''.join([plot_folder, 'tsne_cuda_template'])
        embed_scatter(temp_embedding, filename=filename)

        print("Embedding full dataset onto template")
        from sklearn.neighbors import KNeighborsRegressor

        reembedder = KNeighborsRegressor(n_neighbors=5,weights='distance',n_jobs=12)
        start = time.time()
        reembedder.fit(template,temp_embedding)
        print("Total Time: ", time.time()-start)
        start = time.time()
        final_embedding = reembedder.predict(full_data)
        print("Total Time: ", time.time()-start)

        # from KNNEmbed import KNNEmbed
        # import pdb; pdb.set_trace()
        # reembedder = KNNEmbed(k=5)
        # reembedder.fit(template,temp_embedding)
        # final_embedding = reembedder.predict(full_data)

        filename = ''.join([plot_folder, 'tsne_cuda_final'])
        embed_scatter(final_embedding, filename=filename)
        clustering(final_embedding, filename)

    elif method == 'umap':
        import umap
        umap_transform = umap.UMAP(n_neighbors=100, verbose=True)
        temp_embedding = umap_transform.fit_transform(template)
        filename = ''.join([plot_folder, 'umap_template'])
        embed_scatter(temp_embedding, filename=filename)

        print("Embedding full dataset onto template")
        final_embedding = umap_transform.transform(full_data)
        filename = ''.join([plot_folder, 'umap_final'])
        embed_scatter(final_embedding, filename=filename)
        clustering(final_embedding, filename)
