import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure
import numpy as np

def embed_scatter(data, 
                  filename = './embedding_scatter/',
                  save = True,
                  colorby = None):
    f = plt.figure()
    if colorby is not None:
        color=colorby
    else:
        color = None
    plt.scatter(data[:,0], data[:,1], marker='.', s=3, linewidths=0,
                c=color,cmap='viridis_r', alpha=0.75)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    if colorby is not None:
        plt.colorbar()
    if save:
        plt.savefig(''.join([filename,'.png']),dpi=400)
    plt.close()

def clustering(data, filename=None, bins_per_edge=1000, sigma = 15, max_clip=0.75):
    gauss_filt_hist, x_bin_idx, y_bin_idx = map_density(data,bins_per_edge=bins_per_edge,sigma=sigma,max_clip=max_clip)

    print("Calculating watershed")
    density_thresh = 1e-5
    watershed_map = watershed(-gauss_filt_hist,
                              mask=gauss_filt_hist>density_thresh,
                              watershed_line=False)
    watershed_borders = np.empty((0,2))
    for i in range(1, len(np.unique(watershed_map))):
        contour = measure.find_contours(watershed_map.T==i, 0.5)[0]
        watershed_borders = np.append(watershed_borders, contour,axis=0)
    
    if filename is not None:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(np.log1p(gauss_filt_hist))
        ax.set_aspect('auto')
        plt.savefig(''.join([filename,'_2dhist_gauss.png']),dpi=400)
        plt.close()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(watershed_map)
        ax.set_aspect('auto')
        ax.plot(watershed_borders[:,0],watershed_borders[:,1],'.r',markersize=0.05)
        plt.savefig(''.join([filename,'_watershed.png']),dpi=400)
        plt.close()
    # import pdb; pdb.set_trace()
    data_by_cluster = watershed_map[x_bin_idx.astype(int),y_bin_idx.astype(int)]
    print(str(int(np.amax(data_by_cluster)+1)),"clusters detected")
    print(str(np.unique(data_by_cluster).shape),"unique clusters detected")
    print(np.unique(data_by_cluster))

    return watershed_map, watershed_borders, data_by_cluster, gauss_filt_hist

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
            continue
            # sampled_idx = np.random.choice(np.arange(len(points)), size=size, replace=True)
            # sampled_points = np.append(sampled_points, points[sampled_idx,:], axis=0)
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
        tsne = tc.TSNE(n_iter=2500, verbose=2, num_neighbors=300, perplexity=int(n/100), learning_rate=int(n/12))
        temp_embedding = tsne.fit_transform(template)
        filename = ''.join([plot_folder, 'tsne_cuda_template'])
        embed_scatter(temp_embedding, filename=filename)
        clustering(temp_embedding, filename)

        print("Embedding full dataset onto template")
        from KNNEmbed import KNNEmbed
        start = time.time()
        reembedder = KNNEmbed(k=5)
        reembedder.fit(template,temp_embedding)
        final_embedding = reembedder.predict(full_data, weights='distance')
        print("Total Time ReEmbedding: ", time.time()-start)

        filename = ''.join([plot_folder, 'tsne_cuda_final'])
        embed_scatter(final_embedding, filename=filename)
        _, _, _, density_map = clustering(final_embedding, filename, sigma=50, bins_per_edge=5000)
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


def cluster_frequencies(data_by_cluster, batch_ID):
    '''
        batch_ID: list of metadata label for each element in data_by_cluster
    '''

    num_batches = len(set(batch_ID))
    cluster_freqs = np.zeros((num_batches, np.max(data_by_cluster)+1))
    for i, batch in enumerate(set(batch_ID)):
        cluster_by_ID = data_by_cluster[batch_ID==batch]
        cluster_freqs[i,:] = np.histogram(cluster_by_ID, bins=range(-1, np.max(data_by_cluster)+1))[0]

    frame_totals = np.sum(cluster_freqs,axis=1)
    cluster_freqs = cluster_freqs/np.expand_dims(frame_totals,axis=1) #Fraction of total

    return cluster_freqs