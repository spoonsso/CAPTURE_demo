def sample_by_meta(data, 
                   data_by_cluster: Union[np.array, List[int]], 
                   sample_size: int = 20):
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