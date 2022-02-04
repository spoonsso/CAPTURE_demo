%% Testing PCA vs sklearn incremental PCA

addpath(genpath('/hpc/group/tdunn/joshwu/CAPTURE_demo/'))
flag = int32(bitor(2,8))
py.sys.setdlopenflags(flag);

warning('off','MATLAB:chckxy:IgnoreNaN')

disp("Loading Data")
x = load('fisheriris.mat');
data = x.meas;

py.importlib.import_module("numpy");
py.importlib.import_module("incremental_pca");

[coeffs_m, score_m, latent_m, ~, explained_m] = pca(data);
data_np = py.numpy.array(data(:).');
data_np = data_np.reshape(py.int(size(data,1)), py.int(size(data,2)));
pca_py = py.incremental_pca.incremental_pca(data_np,py.int(2));
score_p = squeeze(double(py.numpy.array(pca_py(2)))).';
coeffs_p = squeeze(double(py.numpy.array(pca_py(2)))).';
explained_p = double(py.numpy.array(pca_py(3))).';