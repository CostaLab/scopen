import time
import argparse

from sklearn.feature_extraction.text import TfidfTransformer
from multiprocessing import Pool, cpu_count
from kneed import KneeLocator

from .MF import NMF
from .Utils import *
from .__version__ import __version__


def parse_args():
    parser = argparse.ArgumentParser(description='scOpen')
    parser.add_argument("--input", type=str, default=None,
                        help="Input name, can be a file name or a directory")
    parser.add_argument("--input_format", type=str, default='dense',
                        help="Input format. Currently available: dense, 10X, 10Xh5, sparse. "
                             "Dense: a text file where each row is a peak and each column represents a cell. "
                             "10X: a folder including barcodes.tsv, matrix.mtx and peaks.bed. "
                             "10Xh5: a h5 file generated by standard cell-ranger pipeline. "
                             "sparse: a text file including three columns, the first column indicates peaks, the second"
                             "columns represent barcodes and the third one is the number of reads. "
                             "Default: dense")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(),
                        help="If specified, all output files will be written to that directory. "
                             "Default: current working directory")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output filename")
    parser.add_argument("--output_format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X, 10Xh5. "
                             "Default: dense")
    parser.add_argument("--n_components", type=int, default=30,
                        help="Number of components. "
                             "Default: 30")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Parameter for model regularization to prevent from over-fitting. "
                             "Default: 1")
    parser.add_argument("--max_iter", type=int, default=500,
                        help="Number of iteration for optimization. "
                             "Default: 500")
    parser.add_argument("--estimate_rank", default=False,
                        action='store_true',
                        help='If set, number of components will be selected by knee point. '
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
    parser.add_argument("--init", type=str, default="nndsvd",
                        help="Method used to initialize the procedure."
                             "Default: nndsvd")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state. Default: 42")
    parser.add_argument("--nc", type=int, metavar="INT", default=1,
                        help="The number of cores. DEFAULT: 1")
    parser.add_argument("--no_impute", default=False,
                        action='store_true',
                        help='If set, scOpen will not save the imputed matrix. '
                             'Set it if you only want to use the dimension reduced matrix. '
                             'Default: False')
    parser.add_argument("--binary", default=False,
                        action='store_true',
                        help='If set, a binary matrix will also be generated. '
                             'Default: False')
    parser.add_argument("--binary_quantile", type=float, default=0.5,
                        help="The quantile value for binarize matrix."
                             "Default: 0.5")
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def scopen_dr(counts, n_components=30, alpha=1, max_iter=200, verbose=1, random_state=42, init='nndsvd'):
    tf_idf = tf_idf_transform(data=counts)
    w_hat, h_hat, err = run_nmf(arguments=(tf_idf, n_components, alpha, max_iter, verbose, random_state, init))

    return h_hat


def tf_idf_transform(data):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running tf-idf transformation...")
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(data))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(data)))

    return tf_idf


def run_nmf(arguments):
    data, n_components, alpha, max_iter, verbose, random_state, init = arguments
    model = NMF(n_components=n_components,
                random_state=random_state,
                init=init,
                alpha=alpha,
                l1_ratio=0,
                max_iter=max_iter,
                verbose=verbose)

    w_hat = model.fit_transform(X=data)
    h_hat = model.components_

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"ranks: {n_components}, fitting error: {model.reconstruction_err_}")

    return [w_hat, h_hat, model.reconstruction_err_]


def estimate_rank(data, args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running rank estimation...")

    n_components_list = np.arange(start=args.min_n_components,
                                  stop=args.max_n_components + 1,
                                  step=args.step_n_components)

    w_hat_dict, h_hat_dict, error_list = dict(), dict(), list()
    if args.nc == 1:
        for n_components in n_components_list:
            arguments = (data, n_components, args.alpha,
                         args.max_iter, args.verbose,
                         args.random_state, args.init)

            res = run_nmf(arguments)
            w_hat_dict[n_components] = res[0]
            h_hat_dict[n_components] = res[1]
            error_list.append(res[2])

    elif args.nc > 1:
        arguments_list = list()
        for n_components in n_components_list:
            arguments = (data, n_components, args.alpha,
                         args.max_iter, args.verbose,
                         args.random_state, args.init)
            arguments_list.append(arguments)

        with Pool(processes=args.nc) as pool:
            res = pool.map(run_nmf, arguments_list)

        for i, n_components in enumerate(n_components_list):
            w_hat_dict[n_components] = res[i][0]
            h_hat_dict[n_components] = res[i][1]
            error_list.append(res[i][2])

    kl = KneeLocator(n_components_list, error_list,
                     S=1.0, curve="convex", direction="decreasing")

    plot_knee(kl, args)

    return w_hat_dict[kl.knee], h_hat_dict[kl.knee]


def main():
    args = parse_args()
    assert args.input is not None, "Please specify an input"
    print(args)

    start = time.time()

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"detected {cpu_count()} cpus, {args.nc} of them are used.")

    data, barcodes, peaks = load_data(args=args)

    # binarize the input matrix
    data[data > 0] = 1
    (m, n) = data.shape

    n_non_zeros = data.count_nonzero()

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"number of peaks: {m}; number of cells {n}")
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"number of non-zeros before imputation: {n_non_zeros}")
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"sparsity: {1 - n_non_zeros / (m * n)}")

    # plot_open_regions_density(data, args)

    # tf-idf
    tf_idf = tf_idf_transform(data)

    if args.estimate_rank:
        w_hat, h_hat = estimate_rank(data=tf_idf, args=args)

    else:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
              f"running NMF...")
        arguments = (tf_idf, args.n_components, args.alpha,
                     args.max_iter, args.verbose,
                     args.random_state, args.init)

        res = run_nmf(arguments)
        w_hat, h_hat = res[0], res[1]

    df = pd.DataFrame(data=w_hat, index=peaks)
    df.to_csv(os.path.join(args.output_dir, "{}_peaks.txt".format(args.output_prefix)), sep="\t")

    df = pd.DataFrame(data=h_hat, columns=barcodes)
    df.to_csv(os.path.join(args.output_dir, "{}_barcodes.txt".format(args.output_prefix)), sep="\t")

    if not args.no_impute:

        m_hat = np.dot(w_hat, h_hat).astype(np.float32)
        m_hat_binary = None

        if args.binary:
            threshold = np.quantile(m_hat, args.binary_quantile)
            m_hat_binary = (m_hat > threshold).astype(np.int8)

        if args.output_format == "sparse":
            filename = os.path.join(args.output_dir, "{}.txt".format(args.output_prefix))
            write_data_to_sparse_file(filename=filename,
                                      data=m_hat,
                                      barcodes=barcodes,
                                      peaks=peaks)
            if m_hat_binary is not None:
                filename = os.path.join(args.output_dir, "{}_binary.txt".format(args.output_prefix))
                write_data_to_sparse_file(filename=filename,
                                          data=m_hat_binary,
                                          barcodes=barcodes,
                                          peaks=peaks)

        elif args.output_format == "dense":
            filename = os.path.join(args.output_dir, "{}.txt".format(args.output_prefix))
            write_data_to_dense_file(filename=filename,
                                     data=m_hat,
                                     barcodes=barcodes,
                                     peaks=peaks)
            if m_hat_binary is not None:
                filename = os.path.join(args.output_dir, "{}_binary.txt".format(args.output_prefix))
                write_data_to_dense_file(filename=filename,
                                         data=m_hat_binary,
                                         barcodes=barcodes,
                                         peaks=peaks)

        elif args.output_format == "10X":
            write_data_to_10x(output_dir=args.output_dir,
                              data=m_hat,
                              barcodes=barcodes,
                              peaks=peaks)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"total time: {h: .2f}h {m: .2f}m {s: .2f}s")


if __name__ == '__main__':
    main()
