import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
import numpy as np

def embed_scatter(data, 
                  filename = './embedding_scatter',
                  save = True,
                  colorby = None):
    f = plt.figure()
    if colorby is not None:
        color=colorby
    else:
        color = None
    plt.scatter(data[:,0], data[:,1], marker='.', s=3, linewidths=0,
                c=color,cmap='viridis_r', alpha=0.75)
    if colorby is not None:
        plt.colorbar()
    if save:
        plt.savefig(''.join([filename,'.png']),dpi=400)
    plt.close()

def clustering(data, filename, bins_per_edge=1000, sigma = 15):
    x_range = int(np.ceil(np.amax(data[:,0])) - np.floor(np.amin(data[:,0])))
    y_range = int(np.ceil(np.amax(data[:,1])) - np.floor(np.amin(data[:,1])))
    hist,xedges,yedges = np.histogram2d(data[:,0], data[:,1], bins=[bins_per_edge, bins_per_edge],
                          range=[[int(np.floor(np.amin(data[:,0]))-x_range/40),int(np.ceil(np.amax(data[:,0]))+x_range/40)],
                                 [int(np.floor(np.amin(data[:,1]))-y_range/40),int(np.ceil(np.amax(data[:,1]))+y_range/40)]],
                          density=False)
    hist = np.rot90(hist)

    assert xedges[0]<xedges[-1] and yedges[0]<yedges[1]

    x_bin_idx = np.zeros(np.shape(data[:,0]))
    y_bin_idx = np.zeros(np.shape(data[:,1]))
    for i in range(len(x_bin_idx)):
        x_bin_idx[i] = 1001-np.argmax(xedges>data[i,0])
        y_bin_idx[i] = 1001-np.argmax(yedges>data[i,1])

    gauss_filt_hist = gaussian_filter(hist, sigma=sigma)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(gauss_filt_hist)
    ax.set_aspect('auto')
    plt.savefig(''.join([filename,'_2dhist_gauss.png']),dpi=400)
    plt.close()

    print("Calculating watershed")
    watershed_map = watershed(-gauss_filt_hist,connectivity=8, watershed_line=True)
    watershed_borders = np.where(watershed_map==0,1,0)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(watershed_borders, cmap='gray_r')
    ax.set_aspect('auto')
    plt.savefig(''.join([filename,'_watershed.png']),dpi=400)
    plt.close()
    data_by_cluster = watershed_map[x_bin_idx.astype(int),y_bin_idx.astype(int)]
    print(str(int(np.amax(data_by_cluster))),"clusters detected")

    return watershed_map, data_by_cluster, gauss_filt_hist

def sample_clusters(data, data_by_cluster, size=20):
    '''
    Equally sampling points from each cluster for a batchmap
    IN:
        data - All of the data in dataset (may be downsampled)
        data_by_cluster - Cluster number for each point in `data`
        size - Number of points to sample from a cluster
    OUT:
        sampled_points - Values of sampled points from `data`
        idx - Index in `data` of sampled points
    '''
    data = np.append(data,np.expand_dims(np.arange(np.shape(data)[0]),axis=1),axis=1)
    sampled_points = np.empty((0,np.shape(data)[1]))
    for cluster_id in np.unique(data_by_cluster):
        points = data[data_by_cluster==cluster_id,:]
        if len(points)==0:
            continue
        elif len(points)<size:
            sampled_idx = np.random.choice(np.arange(len(points)), size=size, replace=True)
            sampled_points = np.append(sampled_points, points[sampled_idx,:], axis=0)
        else:
            num_points = min(len(points),size)
            sampled_points = np.append(sampled_points, 
                                       np.random.permutation(points)[:num_points], 
                                       axis=0)
    print("Number of points sampled")
    print(sampled_points.shape)
    return sampled_points[:,:-1],np.squeeze(sampled_points[:,-1]).astype(int).tolist()

def reembed(template, template_idx, full_data, method='tsne_cuda', plot_folder='./plots/'):
    import time
    print("Finding template embedding using ", method)
    if method == 'tsne_cuda':
        n = np.shape(template)[0]
        import tsnecuda as tc
        tsne = tc.TSNE(n_iter=5000, verbose=2, num_neighbors=300, perplexity=int(n/100), learning_rate=int(n/12))
        temp_embedding = tsne.fit_transform(template)
        filename = ''.join([plot_folder, 'tsne_cuda_template'])
        embed_scatter(temp_embedding, filename=filename)
        clustering(temp_embedding, filename)

        print("Embedding full dataset onto template")
        from KNNEmbed import KNNEmbed
        start = time.time()
        reembedder = KNNEmbed(k=5)
        reembedder.fit(template,temp_embedding)
        final_embedding = reembedder.predict(full_data,weights='distance')
        print("Total Time ReEmbedding: ", time.time()-start)

        filename = ''.join([plot_folder, 'tsne_cuda_final'])
        embed_scatter(final_embedding, filename=filename)
        _, _, density_map = clustering(final_embedding, filename, sigma=50, bins_per_edge=5000)
        save_file = {'template':template, 'template_embedding': temp_embedding, 'template_idx': np.array(template_idx), 'final_density_map': density_map}
        import hdf5storage
        hdf5storage.savemat(''.join([plot_folder,'results.mat']), save_file)
        print("Saving to ", ''.join([plot_folder,'results.mat']))

    elif method == 'umap':
        import umap
        umap_transform = umap.UMAP(n_neighbors=300, verbose=True)
        temp_embedding = umap_transform.fit_transform(template)
        filename = ''.join([plot_folder, 'umap_template'])
        embed_scatter(temp_embedding, filename=filename)

        print("Embedding full dataset onto template")
        final_embedding = umap_transform.transform(full_data)
        filename = ''.join([plot_folder, 'umap_final'])
        embed_scatter(final_embedding, filename=filename)
        clustering(final_embedding, filename)

    return final_embedding, temp_embedding