from DataStruct import DataStruct
import visualization as vis
from embed import Watershed, BatchEmbed
import tsnecuda as tc
import os
import hdf5storage
from sklearn.decomposition import PCA
import numpy as np
from typing import Optional, Union, List, Tuple
import yaml

def params_config(filepath: str):
    '''
    Reads in params_config
    '''
    with open(filepath) as f:
        config_dict = yaml.safe_load(f)

    return config_dict

def params_process(filepath: str):
    '''
    Loads in and processes params to fill in blanks
    '''
    params = params_config(filepath)

    if 'pca_features' not in params:
        params['pca_features'] = False
    
    return params


def load_data(filepath,
              pca_features: bool = False):
    ds = DataStruct(config_path=paths_config)
    ds.load_feats(downsample=downsample)

    if pca_features:
        ds.features = PCA(n_components=60).fit_transform(ds.features)

    ds.load_meta()
    ds.out_path = ''.join([ds.out_path,'/exp_',method,'/'])

    return ds

def run_analysis(params_config: str,
                 paths_config: str):

    params = params_process(params_config)

    ds = load_data(paths_config,
                   params['pca_features'])

    


    


def exp_embed(params_config: str,
              paths_config:str,
              column: str,
              conditions: List[str],
              method: str = 'batch_fitsne',
              load_batch: bool = False,
              skeleton_vids: bool = True,
              downsample: int = 10):

    ds = DataStruct(config_path=paths_config)
    ds.load_feats(downsample=downsample)

    ds.features = PCA(n_components=60).fit_transform(ds.features)

    ds.load_meta()
    ds.out_path = ''.join([ds.out_path,'/exp_',method,'/'])

    import pdb; pdb.set_trace()
    ds_embed = ds[:,ds.data[column].isin(conditions)]

    if method.startswith('batch'):
        if not load_batch:
            ## TODO: probably just combine the batch tsne and umap classes again
            if method == 'batch_cuda_tsne':
                ## Finding template t-SNE embeddings
                batch_embed = BatchEmbed(sampling_n = 20,
                                        n_iter = 1000,
                                        n_neighbors = 150,
                                        perplexity = 50,
                                        lr = 'auto',
                                        batch_method = 'tsne_cuda',
                                        embed_method = 'fitsne',
                                        sigma=14)

            elif method == 'batch_umap':
                batch_embed = BatchEmbed(sampling_n = 20,
                                        n_neighbors = 100,
                                        perplexity = 50,
                                        min_dist = 0.5,
                                        batch_method = 'fitsne',
                                        embed_method = 'umap',
                                        sigma = 14)

            elif method == 'batch_fitsne':
                batch_embed = BatchEmbed(sampling_n = 20,
                                         perplexity = 50,
                                         batch_method = 'fitsne',
                                         embed_method = 'fitsne',
                                         lr = 'auto',
                                         sigma = 14)

            ## Embedding all points on template
            batch_embed.fit(data = ds_embed.features,
                            batch_id = ds.exp_id,
                            save_batchmaps = ds.out_path,
                            save_temp_scatter = ds.out_path)
            batch_embed.save_pickle(ds.out_path)

        else:
            print("Loading old batch analysis")
            if method =='batch_tsne':
                batch_embed = BatchEmbed().load_pickle(''.join([ds.out_path,'batch_embed.p']))
                batch_embed.embed_template(n_neighbors = 200,
                                            perplexity = 70,
                                            lr = 'auto',
                                            save_scatter = ds.out_path)
            elif method == 'batch_umap':
                batch_embed = BatchEmbed().load_pickle(''.join([ds.out_path,'batch_embed.p']))
                batch_embed.embed(n_neighbors = 60,
                                  min_dist = 0.01,
                                  spread = 10)

        
        ds.embed_vals = batch_embed.predict(ds.features)

        temp_struct = ds[batch_embed.temp_idx,:]

        # Calculating watershed and densities of template to compare with 
        ws = Watershed(sigma = 15,
                        n_bins = 1000,
                        max_clip = 1,
                        log_out = True,
                        pad_factor = 0.05)
        temp_struct.data['Cluster'] = ws.fit_predict(data = temp_struct.embed_vals)

        vis.density(ws.density, ws.borders,
                    filepath = ''.join([ds.out_path,'temp_density.png']))
        vis.density_cat(data=temp_struct, column='Condition', watershed=ws, n_col=2,
                        filepath = ''.join([ds.out_path, 'temp_density_by_condition.png']))
    elif method == 'tsne_cuda':
        tsne = tc.TSNE(n_iter=1000, 
                verbose=2,
                num_neighbors=150,
                perplexity=50,
                learning_rate=200)
        ds.embed_vals = tsne.fit_transform(ds.features)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))
    elif method == 'umap':
        import umap
        umap_transform = umap.UMAP(n_neighbors=70, verbose=True)
        ds.embed_vals = umap_transform.fit_transform(ds.features)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))
    elif method == 'fitsne':
        import fitsne
        ds.embed_valsembedding = fitsne.FItSNE(ds.features.astype(np.double),
                                            perplexity=30,
                                            learning_rate='auto').astype(np.float)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))

    

    if skeleton_vids:
        ds.load_connectivity()
        ds.load_pose()
        vis.skeleton_vid3D_cat(ds, 'Cluster', n_skeletons=10)
    

def embedding_pipe(paths_config: str,
                   load_batch: bool = False,
                   method: str = 'batch_fitsne',
                   skeleton_vids: bool = True,
                   downsample: int = 10):
    # Load data
    ds = DataStruct(config_path=paths_config)
    ds.load_feats(downsample=downsample)

    ds.features = PCA(n_components=60).fit_transform(ds.features)
    # import pdb; pdb.set_trace()

    ds.load_meta()
    ds.out_path = ''.join([ds.out_path,'/',method,'/'])

    if not os.path.exists(ds.out_path):
        os.makedirs(ds.out_path)

    if method.startswith('batch'):
        if not load_batch:
            ## TODO: probably just combine the batch tsne and umap classes again
            if method == 'batch_cuda_tsne':
                ## Finding template t-SNE embeddings
                batch_embed = BatchEmbed(sampling_n = 20,
                                        n_iter = 1000,
                                        n_neighbors = 150,
                                        perplexity = 50,
                                        lr = 'auto',
                                        batch_method = 'tsne_cuda',
                                        embed_method = 'fitsne',
                                        sigma=14)

            elif method == 'batch_umap':
                batch_embed = BatchEmbed(sampling_n = 20,
                                        n_neighbors = 100,
                                        perplexity = 50,
                                        min_dist = 0.5,
                                        batch_method = 'fitsne',
                                        embed_method = 'umap',
                                        sigma = 14)

            elif method == 'batch_fitsne':
                batch_embed = BatchEmbed(sampling_n = 20,
                                         perplexity = 50,
                                         batch_method = 'fitsne',
                                         embed_method = 'fitsne',
                                         transform_method = 'knn',
                                         lr = 'auto',
                                         sigma = 14)

            ## Embedding all points on template
            ds.embed_vals = batch_embed.fit_predict(data = ds.features,
                                                    batch_id = ds.exp_id,
                                                    save_batchmaps = ds.out_path)
            batch_embed.save_pickle(ds.out_path)
        else:
            print("Loading old batch analysis")
            if method =='batch_tsne':
                batch_embed = BatchEmbed().load_pickle(''.join([ds.out_path,'batch_embed.p']))
                batch_embed.embed_template(n_neighbors = 200,
                                            perplexity = 70,
                                            lr = 'auto')
            elif method == 'batch_umap':
                batch_embed = BatchEmbed().load_pickle(''.join([ds.out_path,'batch_embed.p']))
                batch_embed.embed_template(n_neighbors = 60,
                                            min_dist = 0.01,
                                            spread = 10)

            ws = Watershed(sigma = 15,
                        n_bins = 1000,
                        max_clip = 1,
                        log_out = True,
                        pad_factor = 0.05)
            ws = ws.fit(data = batch_embed.temp_embedding)
            vis.density(ws.density, ws.borders,
                        filepath = ''.join([ds.out_path,'temp_density_pre.png']))

            ds.embed_vals = batch_embed.predict(data = ds.features)

        # import pdb; pdb.set_trace()
        temp_struct = ds[batch_embed.temp_idx,:]

        # Calculating watershed and densities of template to compare with 
        ws = Watershed(sigma = 15,
                        n_bins = 1000,
                        max_clip = 1,
                        log_out = True,
                        pad_factor = 0.05)
        temp_struct.data['Cluster'] = ws.fit_predict(data = temp_struct.embed_vals)

        vis.density(ws.density, ws.borders,
                    filepath = ''.join([ds.out_path,'temp_density.png']))
        vis.density_cat(data=temp_struct, column='Condition', watershed=ws, n_col=2,
                        filepath = ''.join([ds.out_path, 'temp_density_by_condition.png']))

    elif method == 'tsne_cuda':
        tsne = tc.TSNE(n_iter=1000, 
                verbose=2,
                num_neighbors=150,
                perplexity=50,
                learning_rate=200)
        ds.embed_vals = tsne.fit_transform(ds.features)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))
    elif method == 'umap':
        import umap
        umap_transform = umap.UMAP(n_neighbors=70, verbose=True)
        ds.embed_vals = umap_transform.fit_transform(ds.features)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))
    elif method == 'fitsne':
        import fitsne
        ds.embed_vals = fitsne.FItSNE(ds.features.astype(np.double),
                                        perplexity=50,
                                        learning_rate='auto').astype(np.float)
        vis.scatter(ds.embed_vals, filepath=''.join([ds.out_path,'scatter.png']))

    # Calculating watershed
    ws = Watershed(sigma = 14,
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

    if skeleton_vids:
        ds.load_connectivity()
        ds.load_pose()
        vis.skeleton_vid3D_cat(ds, 'Cluster', n_skeletons=10)


# embedding_pipe(paths_config='../configs/path_configs/embedding_analysis_ws_r01.yaml',
#                method='batch_fitsne',
#                downsample = 10,
#                skeleton_vids=False)

# embedding_pipe(paths_config='../configs/path_configs/embedding_analysis_ws_48.yaml',
#                method='fitsne',
#                downsample = 1,
#                skeleton_vids = None)

embedding_pipe(paths_config='../configs/path_configs/embedding_analysis_ws_48.yaml',
               method='batch_fitsne',
               downsample = 1,
               skeleton_vids = None)

# embedding_pipe(paths_config='../embedding_analysis_ws_48.yaml',
#                method='batch_umap', load_batch = False,
#                downsample = 1)

# embedding_pipe(paths_config='../embedding_analysis_ws_60.yaml',
#                method='umap',
#                downsample = 20)


