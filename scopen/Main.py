import time
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import check_array
from multiprocessing import Pool, cpu_count
from kneed import KneeLocator

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
    parser.add_argument("--n_neighbors", type=int, default=5,
                        help="Number of neighbors used for knn imputation. "
                             "Default: 5")
    parser.add_argument("--select_model", default=False,
                        action='store_true',
                        help='If set, both number of components and alpha will be determined by using cross validation.'
                             'Default: False')
    parser.add_argument("--min_n_components", type=int, default=2,
                        help="Minimum of number of components for model selection, must be positive."
                             "Default: 2")
    parser.add_argument("--max_n_components", type=int, default=30,
                        help="Minimum of number of components for model selection, must be positive."
                             "Default: 2")
    parser.add_argument("--step_n_components", type=int, default=1,
                        help="Spacing between values"
                             "Default: 1")
    parser.add_argument("--alpha_list", default='0.001,0.01,0.1,1',
                        help='Alpha list for model selection.'
                             'Default: 0.001, 0.01, 0.1, 1')
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion of data for model selection, should be between 0 and 1."
                             "Default: 0.1")
    parser.add_argument("--random_state", type=int, default=42, help="Random state. Default: 42")
    parser.add_argument("--nc", type=int, metavar="INT", default=1,
                        help="The number of cores. DEFAULT: 1")
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def knn_impute(data, n_neighbors):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running knn imputation...")

    neigh = NearestNeighbors(n_neighbors=n_neighbors,
                             n_jobs=-1,
                             metric='cosine')
    neigh.fit(np.transpose(data))
    indices = neigh.kneighbors(return_distance=False)

    data_impute = data.copy()

    for i in range(indices.shape[0]):
        _y = np.sum(data[:, indices[i]], axis=1)

        idx = (data[:, i] == 0) & (_y > 0)
        # set missing elements to NA
        data_impute[idx, i] = np.nan

    return data_impute


def tf_idf_transform(data):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running tf-idf transformation...")
    model = TfidfTransformer(smooth_idf=False)
    tf_idf = np.transpose(model.fit_transform(np.transpose(data)))

    if sp_sparse.issparse(tf_idf):
        tf_idf = tf_idf.toarray()

    return tf_idf


def run_nmf(data, n_components, alpha, max_iter, verbose, random_state,
            beta_loss, w=None, h=None):
    if np.any(np.isnan(data)):
        solver, init = "mu", "random"
    else:
        solver, init = "cd", "nndsvd"

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

    return w_hat, h_hat, model.reconstruction_err_


def select_model(data, args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running model selection...")

    n_components_list = np.arange(start=args.min_n_components,
                                  stop=args.max_n_components + 1,
                                  step=args.step_n_components)

    w_hat_dict, h_hat_dict, error_list = dict(), dict(), list()
    for n_components in n_components_list:
        w_hat, h_hat, _error = run_nmf(data=data,
                                       n_components=n_components,
                                       alpha=args.alpha,
                                       max_iter=args.max_iter,
                                       verbose=args.verbose,
                                       random_state=args.random_state,
                                       beta_loss="frobenius")

        w_hat_dict[n_components] = w_hat
        h_hat_dict[n_components] = h_hat
        error_list.append(_error)

        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
              f"ranks: {n_components}, training error: {_error}")

    kl = KneeLocator(n_components_list, error_list,
                     S=1.0, curve="convex", direction="decreasing")

    plot_knee(kl, args)

    return w_hat_dict[kl.knee], h_hat_dict[kl.knee]


def main():
    start = time.time()

    args = parse_args()
    print(args)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"detected {cpu_count()} cpus, {args.nc} of them are used.")

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

    # tf-idf
    tf_idf = tf_idf_transform(data)

    if args.knn_impute:
        tf_idf = knn_impute(data=tf_idf, n_neighbors=args.n_neighbors)

    if args.select_model:
        w_hat, h_hat = select_model(data=tf_idf, args=args)

    else:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
              f"running NMF...")
        w_hat, h_hat, _error = run_nmf(data=tf_idf,
                                       n_components=args.n_components,
                                       alpha=args.alpha,
                                       max_iter=args.max_iter,
                                       verbose=args.verbose,
                                       random_state=args.random_state,
                                       beta_loss="frobenius")

    save_data(w=w_hat, h=h_hat, peaks=peaks, barcodes=barcodes, args=args)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"total time: {h: .2f}h {m: .2f}m {s: .2f}s")


if __name__ == '__main__':
    main()
