import time
import argparse

from .MF import bounded_non_negative_factorization
from .Utils import *
from .__version__ import __version__


def parse_args():
    parser = argparse.ArgumentParser(description='scOpen')
    parser.add_argument("--input", type=str, default=None,
                        help="Input name, can be a file name or a directory")
    parser.add_argument("--input-format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X."
                             "Default: dense")
    parser.add_argument("--output-dir", type=str, default=os.getcwd(),
                        help="If specified, all output files will be written to that directory. "
                             "Default: current working directory")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output filename")
    parser.add_argument("--output-format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X, 10Xh5."
                             "Default: dense")
    parser.add_argument("--n-components", type=int, default=30, help="Number of components. Default: 30")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--min-rho", type=float, default=0.0)
    parser.add_argument("--max-rho", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--decimals", type=int, default=8)
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def main():
    args = parse_args()

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

    print("Number of peaks: {}; number of cells {}".format(m, n))
    print("Sparsity before imputation: {}".format(1 - np.count_nonzero(data) / (m * n)))

    n_open_regions = np.log10(data.sum(axis=0))
    max_n_open_regions = np.max(n_open_regions)
    min_n_open_regions = np.min(n_open_regions)

    rho = args.min_rho + (args.max_rho - args.min_rho) * \
          (max_n_open_regions - n_open_regions) / (max_n_open_regions - min_n_open_regions)

    data = data[:, :] * (1 / (1 - rho))

    w_hat, h_hat, _ = bounded_non_negative_factorization(X=data,
                                                         n_components=args.n_components,
                                                         alpha=args.alpha,
                                                         max_iter=args.max_iter,
                                                         verbose=args.verbose)
    del data
    m_hat = np.dot(w_hat, h_hat)
    np.clip(m_hat, 0, 1, out=m_hat)
    np.round(m_hat, decimals=args.decimals, out=m_hat)

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

    # elif args.output_format == "10X":
    #     filename = os.path.join(args.output_dir, "{}.h5".format(args.output_prefix))
    #
    #     with h5py.File(output_filename, 'w') as f:
    #         group = f.create_group('matrix')
    #         group['name'] = peaks
    #         group['barcodes'] = columns
    #         group['data'] = m_hat

    print("Sparsity after imputation: {}".format(1 - np.count_nonzero(m_hat) / (m * n)))
    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print("[total time: ", "%dh %dm %ds" % (h, m, s), "]")


if __name__ == '__main__':
    main()
