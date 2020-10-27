import time
import argparse
from sklearn.neighbors import NearestNeighbors

from .MF import non_negative_factorization
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
    if args.input_format == "sparse":
        data, barcodes, peaks = get_data_from_sparse_file(filename=args.input)

    elif args.input_format == "dense":
        data, barcodes, peaks = get_data_from_dense_file(filename=args.input)

    elif args.input_format == "10Xh5":
        data, barcodes, peaks = get_data_from_10x_h5(filename=args.input)

    elif args.input_format == "10X":
        data, barcodes, peaks = get_data_from_10x(input_dir=args.input)

    data = np.greater(data, 0)
    (m, n) = data.shape

    print(f"Number of peaks: {m}; number of cells {n}")
    print(f"Number of non-zeros before imputation: {np.count_nonzero(data)}")
    print(f"Sparsity: {1 - np.count_nonzero(data) / (m * n)}")

    n_open_regions = np.count_nonzero(data, axis=0)
    plot_open_regions_density(n_open_regions, args)

    if args.rho is None:
        print(f"Estimating dropout rate...")
        rho, data_y = compute_rho_by_knn(data, k=args.n_neighbors)
        filename = os.path.join(args.output_dir, "{}_y.txt".format(args.output_prefix))
        write_data_to_dense_file(filename=filename, data=data_y, barcodes=barcodes, peaks=peaks)

        plot_estimated_dropout(rho=rho, args=args)
        print(f"Estimate of dropout rate is done!")

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
        # create test data
        non_zero_idx = np.where(data > 0)

        for n_components in [5, 10, 15, 20, 25, 30]:
            w_hat, h_hat, obj_list = non_negative_factorization(X=data,
                                                                n_components=n_components,
                                                                alpha=args.alpha,
                                                                max_iter=args.max_iter,
                                                                verbose=args.verbose)
            res = np.square(data - np.dot(w_hat, h_hat)).mean()
            print(f"rank: {n_components}, residual : {res}")

    else:
        w_hat, h_hat, obj_list = non_negative_factorization(X=data,
                                                            n_components=args.n_components,
                                                            alpha=args.alpha,
                                                            max_iter=args.max_iter,
                                                            verbose=args.verbose)

        plot_objective(obj_list, args)

    del data
    m_hat = np.dot(w_hat, h_hat)
    np.clip(m_hat, 0, 1, out=m_hat)

    if args.binarize:
        threshold = np.quantile(m_hat, args.quantile, axis=0)
        m_hat = (m_hat > threshold).astype(int)

    df = pd.DataFrame(data=w_hat, index=peaks)
    df.to_csv(os.path.join(args.output_dir, "{}_peaks.txt".format(args.output_prefix)), sep="\t")

    df = pd.DataFrame(data=h_hat, columns=barcodes)
    df.to_csv(os.path.join(args.output_dir, "{}_barcodes.txt".format(args.output_prefix)), sep="\t")

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
