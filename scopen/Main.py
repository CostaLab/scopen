import time
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import check_array

from .MF import NMF, _beta_divergence
from .Utils import *
from .__version__ import __version__


def parse_args():
    parser = argparse.ArgumentParser(description='scOpen')
    parser.add_argument("--input", type=str, default=None,
                        help="Input name, can be a file name or a directory")
    parser.add_argument("--input_format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X."
                             "Default: dense")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(),
                        help="If specified, all output files will be written to that directory. "
                             "Default: current working directory")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output filename")
    parser.add_argument("--output_format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X, 10Xh5."
                             "Default: dense")
    parser.add_argument("--n_components", type=int, default=30, help="Number of components. Default: 30")
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--knn_impute", default=False, action='store_true',
                        help="If set, the matrix will first be imputed by using k-nearest neighbors"
                             "Default: False")
    parser.add_argument("--n_neighbors", type=int, default=1,
                        help="Number of neighbors used for knn imputation. "
                             "Default: 1")
    parser.add_argument("--select_model", default=False,
                        action='store_true',
                        help='If set, both number of components and alpha will be determined by using cross validation.'
                             'Default: False')
    parser.add_argument("--n_components_list", default='5,6,7,8,9,10,15,20,25,30',
                        help='Components number list for model selection.'
                             'Default: 5, 6, 7, 8, 9, 10, 15, 20, 25, 30')
    parser.add_argument("--alpha_list", default='0.001,0.01,0.1,1',
                        help='Alpha list for model selection.'
                             'Default: 0.001, 0.01, 0.1, 1')
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion of data for model selection, should be between 0 and 1."
                             "Default: 0.1")
    parser.add_argument("--random_state", type=int, default=42, help="Random state. Default: 42")
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def knn_impute(data, n_neighbors):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"applying knn imputation...")

    model = TfidfTransformer(smooth_idf=False)
    tf_idf = model.fit_transform(np.transpose(data))

    neigh = NearestNeighbors(n_neighbors=n_neighbors,
                             n_jobs=-1,
                             metric='cosine')
    neigh.fit(tf_idf)
    indices = neigh.kneighbors(return_distance=False)

    data_y = np.zeros(data.shape, dtype=np.int)
    for i in range(indices.shape[0]):
        _y = np.sum(data[:, indices[i]], axis=1)
        data_y[:, i] = np.greater(data[:, i] + _y, 0)

    return data_y


def tf_idf_transform(data):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"applying tf-idf transformation...")
    model = TfidfTransformer(smooth_idf=False)
    tf_idf = np.transpose(model.fit_transform(np.transpose(data)))

    if sp_sparse.issparse(tf_idf):
        tf_idf = tf_idf.toarray()

    return tf_idf


def train_test_split(data, args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"splitting data for cross validation...")
    m, n = data.shape

    # create test data by setting 10% elements to NAN
    test_mask = np.empty(shape=(m, n), dtype=np.bool)

    # set seed for reproducibility
    np.random.seed(args.random_state)
    for i in range(m):
        test_mask[i, :] = np.random.choice([True, False], n, p=[args.test_size, 1 - args.test_size])

    data_train, data_test, = data.copy(), data.copy()
    data_train[test_mask] = np.nan
    data_test[~test_mask] = np.nan

    data_train = check_array(data_train, accept_sparse=('csr', 'csc'), dtype=float,
                             force_all_finite=False)

    data_test = check_array(data_test, accept_sparse=('csr', 'csc'), dtype=float,
                            force_all_finite=False)

    num_train = np.count_nonzero(~np.isnan(data_train))
    num_test = np.count_nonzero(~np.isnan(data_test))

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"training data: {num_train}, test data: {num_test}")

    return data_train, data_test


def run_nmf(data, n_components, alpha, max_iter, verbose, random_state,
            beta_loss, solver, init=None, w=None, h=None):
    model = NMF(n_components=n_components,
                random_state=random_state,
                alpha=alpha,
                l1_ratio=0,
                max_iter=max_iter,
                verbose=verbose,
                beta_loss=beta_loss,
                solver=solver,
                init=init)

    w_hat = model.fit_transform(X=data, W=w, H=h)
    h_hat = model.components_

    return w_hat, h_hat


def select_model(data, args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running model selection...")

    data_train, data_test = train_test_split(data=data, args=args)

    w_best, h_best, rank_best, alpha_best = None, None, None, None
    best_error = np.inf

    n_components_list = list(map(int, args.n_components_list.split(',')))
    alpha_list = list(map(float, args.alpha_list.split(',')))

    print(n_components_list)
    print(alpha_list)

    for n_components in n_components_list:
        for alpha in alpha_list:
            w_hat, h_hat = run_nmf(data=data_train,
                                   n_components=n_components,
                                   alpha=alpha,
                                   max_iter=args.max_iter,
                                   verbose=args.verbose,
                                   random_state=args.random_state,
                                   beta_loss="frobenius",
                                   solver="mu",
                                   init="random")

            train_error = _beta_divergence(data_train, w_hat, h_hat,
                                           beta='frobenius',
                                           square_root=True)

            test_error = _beta_divergence(data_test, w_hat, h_hat,
                                          beta='frobenius',
                                          square_root=True)

            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
                  f"ranks: {n_components}, alpha: {alpha}, "
                  f"training error: {train_error}, test error: {test_error}")

            if test_error < best_error:
                best_error = test_error
                w_best, h_best, rank_best, alpha_best = w_hat, h_hat, n_components, alpha

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"model selection done, best rank: {rank_best}, best alpha: {alpha_best}")

    return w_best, h_best, rank_best, alpha_best


def main():
    start = time.time()

    args = parse_args()
    print(args)

    data, barcodes, peaks = load_data(args=args)

    data = np.greater(data, 0)
    (m, n) = data.shape

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"number of peaks: {m}; number of cells {n}")
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"number of non-zeros before imputation: {np.count_nonzero(data)}")
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"sparsity: {1 - np.count_nonzero(data) / (m * n)}")

    plot_open_regions_density(data, args)

    if args.knn_impute:
        data = knn_impute(data=data, n_neighbors=args.n_neighbors)

    # tf-idf
    tf_idf = tf_idf_transform(data)

    if args.select_model:
        w_hat, h_hat, rank_best, alpha_best = select_model(data=tf_idf,
                                                           args=args)

    else:
        w_hat, h_hat = run_nmf(data=tf_idf,
                               n_components=args.n_components,
                               alpha=args.alpha,
                               max_iter=args.max_iter,
                               verbose=args.verbose,
                               random_state=args.random_state,
                               beta_loss="frobenius",
                               solver="cd")

    save_data(w=w_hat, h=h_hat, peaks=peaks, barcodes=barcodes, args=args)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print(f"Done, total time: {h: .2f}h {m: .2f}m {s: .2f}s")


if __name__ == '__main__':
    main()
