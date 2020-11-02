import time
from datetime import datetime
import argparse
from sklearn.neighbors import NearestNeighbors

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
    parser.add_argument("--n_neighbors", type=int, default=1, help="Number of neighbors used for dropout calculation. "
                                                                   "Default: 1")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--rho", type=float, default=None,
                        help='If set, will use this number as dropout rate.'
                             'Default: None')
    parser.add_argument("--binarize", default=False,
                        action='store_true',
                        help='If set, the imputed matrix will be sparsified based on thresholds by keeping quantile.'
                             'Default: False')
    parser.add_argument("--estimate_rank", default=False,
                        action='store_true',
                        help='If set, the number of components will be determined by using cross validation.'
                             'Default: False')
    parser.add_argument("--quantile", type=float, default=None,
                        help='If set, will use this number to binarize the imputed matrix.'
                             'must be between 0 and 1. Default: None')
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def compute_rho_by_knn(data, k):
    neigh = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1, metric='jaccard')
    neigh.fit(np.transpose(data))
    indices = neigh.kneighbors(return_distance=False)

    data_y = np.zeros(data.shape)
    rho = np.zeros(indices.shape[0])
    for i in range(indices.shape[0]):
        _y = np.greater(np.sum(data[:, indices[i]], axis=1), 0)
        data_y[:, i] = np.greater(data[:, i] + _y, 0)
        rho[i] = (np.count_nonzero(data_y[:, i]) - np.count_nonzero(data[:, i])) / np.count_nonzero(data_y[:, i])

    return rho, data_y


def main():
    args = parse_args()

    if args.binarize:
        assert args.quantile is not None, "please provide a number to compute the q-th quantile"

    start = time.time()

    data, barcodes, peaks = None, None, None

    print(args)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, loading data...")
    if args.input_format == "sparse":
        data, barcodes, peaks = get_data_from_sparse_file(filename=args.input)

    elif args.input_format == "dense":
        data, barcodes, peaks = get_data_from_dense_file(filename=args.input)

    elif args.input_format == "10Xh5":
        data, barcodes, peaks = get_data_from_10x_h5(filename=args.input)

    elif args.input_format == "10X":
        data, barcodes, peaks = get_data_from_10x(input_dir=args.input)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, data loaded...")

    data = np.greater(data, 0)
    (m, n) = data.shape

    print(f"Number of peaks: {m}; number of cells {n}")
    print(f"Number of non-zeros before imputation: {np.count_nonzero(data)}")
    print(f"Sparsity: {1 - np.count_nonzero(data) / (m * n)}")

    n_open_regions = np.count_nonzero(data, axis=0)
    plot_open_regions_density(n_open_regions, args)

    if args.rho is None:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, estimating dropout rate...")
        rho, data_y = compute_rho_by_knn(data, k=args.n_neighbors)
        filename = os.path.join(args.output_dir, "{}_y.txt".format(args.output_prefix))
        write_data_to_dense_file(filename=filename, data=data_y, barcodes=barcodes, peaks=peaks)

        plot_estimated_dropout(rho=rho, args=args)
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, dropout rate estimated!")

        df = pd.DataFrame({'num_peaks': n_open_regions,
                           'estimated_dropout': rho,
                           'barcodes': barcodes})

        filename = os.path.join(args.output_dir, "{}_stat.txt".format(args.output_prefix))
        df.to_csv(filename, index=False, sep="\t")

    else:
        rho = args.rho

    data = data[:, :] * (1 / (1 - rho))

    filename = os.path.join(args.output_dir, "{}_x.txt".format(args.output_prefix))
    write_data_to_dense_file(filename=filename, data=data, barcodes=barcodes, peaks=peaks)

    if args.estimate_rank:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, estimating ranks...")
        # create test data by setting 10% elements to NAN
        test_mask = np.zeros(shape=(m, n), dtype=np.bool)
        for i in range(m):
            test_mask[i, :] = np.random.choice([True, False], n, p=[0.1, 0.9])

        data_train = data.copy()
        data_train[test_mask] = np.nan

        num_train = np.count_nonzero(~np.isnan(data_train))
        num_test = np.count_nonzero(np.isnan(data_train))

        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
              f"training data: {num_train}, test data: {num_test}")

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

            train_error = _beta_divergence(np.ma.masked_array(data, test_mask),
                                           w_hat, h_hat, beta=2,
                                           square_root=True)

            test_error = _beta_divergence(np.ma.masked_array(data, ~test_mask),
                                          w_hat, h_hat, beta=2, square_root=True)

            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
                  f"ranks: {n_components}, training error: {train_error}, test error: {test_error}")

            train_error_list.append(train_error)
            test_error_list.append(test_error)

            if test_error < best_error:
                # check if any cell has zero column-sum
                column_sum = np.dot(w_hat, h_hat).sum(axis=0)
                if np.count_nonzero(column_sum) == len(column_sum):
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

        best_w_hat = model.fit_transform(data)
        best_h_hat = model.components_

    del data
    m_hat = np.dot(best_w_hat, best_h_hat)
    np.clip(m_hat, 0, 1, out=m_hat)

    if args.binarize:
        threshold = np.quantile(m_hat, args.quantile, axis=0)
        m_hat = (m_hat > threshold).astype(int)

    output_wh(best_w_hat, best_h_hat, peaks, barcodes, args)

    if args.output_format == "sparse":
        filename = os.path.join(args.output_dir, "{}.txt".format(args.output_prefix))
        write_data_to_sparse_file(filename=filename, data=m_hat, barcodes=barcodes, peaks=peaks)

    elif args.output_format == "dense":
        filename = os.path.join(args.output_dir, "{}.txt".format(args.output_prefix))
        write_data_to_dense_file(filename=filename, data=m_hat, barcodes=barcodes, peaks=peaks)

    elif args.output_format == "10X":
        write_data_to_10x(output_dir=args.output_dir, data=m_hat, barcodes=barcodes, peaks=peaks)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print("[total time: ", "%dh %dm %ds" % (h, m, s), "]")


if __name__ == '__main__':
    main()
