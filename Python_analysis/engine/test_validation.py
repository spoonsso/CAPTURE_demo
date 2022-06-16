import embed
import DataStruct as ds
import validation as val
import visualization as vis

import numpy as np
import pandas as np
import matplotlib.pyplot as plt

import pickle

an_dir = '../results/24_final/batch_fitsne_filter/'

embedder = embed.BatchEmbed().load_pickle(''.join([an_dir,'batch_embed.p']))
ds = pickle.load(open(''.join([an_dir,'datastruct.p']),'rb'))

kf = val.KFoldEmbed(out_path = ds.out_path).run(embedder,
                                            param = 'k',
                                            param_range = [1,3,5,7,10,15,25,50])
kf.plot_error()

embedder = embed.BatchEmbed().load_pickle(''.join([an_dir,'batch_embed.p']))
embedder.transform_method = 'xgboost'
kf = val.KFoldEmbed(out_path = ds.out_path).run(embedder,
                                            param = 'n_trees',
                                            param_range = [10,25,50,75,100,250,500])

kf.plot_error()

