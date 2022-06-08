from DataStruct import DataStruct
import visualization as vis
from embed import BatchTSNE, Watershed
import tsnecuda as tc
import os


def embedding_pipe(config_path: str,
                   load_batch: bool = False,
                   method: str = 'batch_tsne',
                   skeleton_vids: bool = True,
                   downsample: int = 10):
    # Load data
    ds = DataStruct(config_path=config_path)
    ds.load_feats(downsample=downsample)
    ds.load_meta()
    ds.out_path = ''.join([ds.out_path,'/',method,'/'])

    if not os.path.exists(ds.out_path):
        os.makedirs(ds.out_path)

    if method == 'batch_tsne':
        if not load_batch:
            ## Finding template t-SNE embeddings
            bt = BatchTSNE(sampling_n = 20,
                        n_iter = 1000,
                        n_neighbors = 150,
                        perplexity = 50,
                        lr = 'auto',
                        method = 'tsne_cuda',
                        sigma=16)
            ## Embedding all points on template
            ds.embed_vals = bt.fit_predict(data = ds.features,
                                        batch_id = ds.exp_id,
                                        save_batchmaps = ds.out_path,
                                        save_temp_scatter = ds.out_path)
            bt.save_pickle(ds.out_path)
        else:
            bt = BatchTSNE().load_pickle(''.join([ds.out_path,'batch_tsne.p']))
            bt.embed_template(n_neighbors = 200,
                            perplexity = 70,
                            lr = 'auto',
                            save_scatter = ds.out_path)
            ds.embed_vals = bt.predict(data = ds.features)
    elif method == 'cuda_tsne':
        tsne = tc.TSNE(n_iter=1000, 
                verbose=2,
                num_neighbors=150,
                perplexity=50,
                learning_rate=200)
        ds.embed_vals = tsne.fit_transform(ds.features)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))
    elif method == 'umap':
        import umap
        umap_transform = umap.UMAP(n_neighbors=50, verbose=True)
        ds.embed_vals = umap_transform.fit_transform(ds.features)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))

    # Calculating watershed
    ws = Watershed(sigma = 15,
                   n_bins = 1000,
                   max_clip = 1,
                   log_out = True,
                   pad_factor = 0.05)
    ds.data['Cluster'] = ws.fit_predict(data = ds.embed_vals)
    ds.watershed = ws

    vis.density(ws.density, ws.borders,
                filepath = ''.join([ds.out_path,'final_density.png']))
    vis.density_cat(data=ds, column='Condition', watershed=ws, n_col=2,
                    filepath = ''.join([ds.out_path, 'density_by_condition.png']))

    vis.scatter_on_watershed(data=ds, watershed=ws, column='Cluster')

    ds.write_pickle()

    ds.load_connectivity()
    ds.load_pose()
    vis.skeleton_vid3D_cat(ds, 'Cluster', n_skeletons=10)



# embedding_pipe(config_path='../embedding_analysis_ws_r01.yaml',
#                method='batch_tsne',
#                downsample = 10)

# embedding_pipe(config_path='../embedding_analysis_ws_r01.yaml',
#                method='cuda_tsne',
#                downsample = 10)

embedding_pipe(config_path='../embedding_analysis_ws_60.yaml',
               method='batch_tsne',
               downsample = 20)

embedding_pipe(config_path='../embedding_analysis_ws_60.yaml',
               method='cuda_tsne',
               downsample = 20)

# embedding_pipe(config_path='../embedding_analysis_ws_r01.yaml',
#                method='umap',
#                downsample = 10)

# embedding_pipe(config_path='../embedding_analysis_ws_60.yaml',
#                method='umap',
#                downsample = 20)


