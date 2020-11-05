import time
from datetime import datetime
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
    parser.add_argument("--output_prefix", type=str, default=None, help="Output filename")
    parser.add_argument("--output_format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X, 10Xh5."
                             "Default: dense")
    parser.add_argument("--n_components", type=int, default=30, help="Number of components. Default: 30")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=42, help="Random state. Default: 42")
    parser.add_argument("--knn_impute", default=False, action='store_true',
                        help="If set, the matrix will first be imputed by using k-nearest neighbors"
                             "Default: False")
    parser.add_argument("--n_neighbors", type=int, default=1, help="Number of neighbors used for knn imputation. "
                                                                   "Default: 1")
    parser.add_argument("--estimate_rank", default=False,
                        action='store_true',
                        help='If set, the number of components will be determined by using cross validation.'
                             'Default: False')
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def knn_impute(data, n_neighbors):
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


def train_test_split(data, args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, splitting data for cross validation...")
    m, n = data.shape

    # create test data by setting 10% elements to NAN
    test_mask = np.empty(shape=(m, n), dtype=np.bool)

    # set seed for reproducibility
    np.random.seed(args.random_state)
    for i in range(m):
        test_mask[i, :] = np.random.choice([True, False], n, p=[0.1, 0.9])

    data_train, data_test, = data.copy(), data.copy()
    data_train[test_mask] = np.nan
    data_test[~test_mask] = np.nan

    data_train = check_array(data_train, accept_sparse=('csr', 'csc'), dtype=float,
                             force_all_finite=False)

    data_test = check_array(data_test, accept_sparse=('csr', 'csc'), dtype=float,
                            force_all_finite=False)

    return data_train, data_test


def select_model(n_components_list, alpha_list, data, args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running model selection...")

    data_train, data_test = train_test_split(data=data, args=args)

    num_train = np.count_nonzero(~np.isnan(data_train))
    num_test = np.count_nonzero(~np.isnan(data_test))

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"training data: {num_train}, test data: {num_test}")


def main():
    args = parse_args()

    start = time.time()

    print(args)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, loading data...")
    data, barcodes, peaks = load_data(args=args)
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, data loaded...")

    data = np.greater(data, 0)
    (m, n) = data.shape

    print(f"Number of peaks: {m}; number of cells {n}")
    print(f"Number of non-zeros before imputation: {np.count_nonzero(data)}")
    print(f"Sparsity: {1 - np.count_nonzero(data) / (m * n)}")

    n_open_regions = np.count_nonzero(data, axis=0)
    plot_open_regions_density(n_open_regions, args)

    if args.knn_impute:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, applying knn imputation...")
        data = knn_impute(data=data, n_neighbors=args.n_neighbors)
        filename = os.path.join(args.output_dir, "{}_knn.txt".format(args.output_prefix))
        write_data_to_dense_file(filename=filename, data=data, barcodes=barcodes, peaks=peaks)
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, knn imputation applied!")

    # tf-idf
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, applying tf-idf transformation...")
    model = TfidfTransformer(smooth_idf=False)
    tf_idf = np.transpose(model.fit_transform(np.transpose(data)))

    if sp_sparse.issparse(tf_idf):
        tf_idf = tf_idf.toarray()

    filename = os.path.join(args.output_dir, "{}_tf_idf.txt".format(args.output_prefix))
    write_data_to_dense_file(filename=filename, data=tf_idf, barcodes=barcodes, peaks=peaks)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, tf-idf transformation applied...")

    if args.estimate_rank:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, estimating ranks...")
        # create test data by setting 10% elements to NAN
        test_mask = np.empty(shape=(m, n), dtype=np.bool)

        n_components_list = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
        train_error_list, test_error_list = [], []
        best_error, best_rank = np.inf, None
        best_w_hat, best_h_hat = None, None

        for n_components in n_components_list:
            model = NMF(n_components=n_components,
                        random_state=args.random_state,
                        alpha=args.alpha,
                        l1_ratio=0,
                        max_iter=args.max_iter,
                        verbose=args.verbose,
                        beta_loss="frobenius",
                        solver="mu",
                        init="random")

            w_hat = model.fit_transform(data_train)
            h_hat = model.components_

            train_error = _beta_divergence(data_train,
                                           w_hat, h_hat, beta='frobenius',
                                           square_root=True)

            test_error = _beta_divergence(data_test,
                                          w_hat, h_hat, beta='frobenius',
                                          square_root=True)

            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
                  f"ranks: {n_components}, training error: {train_error}, test error: {test_error}")

            train_error_list.append(train_error)
            test_error_list.append(test_error)

            if test_error < best_error:
                best_error = test_error
                best_w_hat = w_hat
                best_h_hat = h_hat
                best_rank = n_components

        plot_error(n_components_list,
                   train_error_list,
                   test_error_list,
                   best_rank,
                   args)

        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, ranks estimated, best rank: {best_rank}")

    else:
        model = NMF(n_components=args.n_components,
                    random_state=args.random_state,
                    alpha=args.alpha,
                    l1_ratio=0,
                    max_iter=args.max_iter,
                    verbose=args.verbose,
                    beta_loss="frobenius",
                    solver="cd")

        best_w_hat = model.fit_transform(tf_idf)
        best_h_hat = model.components_

    save_data(w=best_w_hat, h=best_h_hat,
              peaks=peaks, barcodes=barcodes, args=args)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print("[total time: ", "%dh %dm %ds" % (h, m, s), "]")


if __name__ == '__main__':
    main()
