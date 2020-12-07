import time
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer
from multiprocessing import Pool, cpu_count
from kneed import KneeLocator
import scipy.sparse as sp
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES

from .MF import NMF
from .Utils import *
from .__version__ import __version__


def parse_args():
    parser = argparse.ArgumentParser(description='scOpen')
    parser.add_argument("--input", type=str, default=None,
                        help="Input name, can be a file name or a directory")
    parser.add_argument("--input_format", type=str, default='dense',
                        help="Input format. Currently available: sparse, dense, 10X. "
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
    parser.add_argument("--alpha", type=float, default=1,
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
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state. Default: 42")
    parser.add_argument("--nc", type=int, metavar="INT", default=1,
                        help="The number of cores. DEFAULT: 1")
    parser.add_argument("--verbose", type=int, default=0)

    version_message = "Version: " + str(__version__)
    parser.add_argument('--version', action='version', version=version_message)

    return parser.parse_args()


def tf_idf_transform(data):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"running tf-idf transformation...")
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(data))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(data)))

    return tf_idf.toarray()


def run_nmf(arguments):
    data, n_components, alpha, max_iter, verbose, random_state = arguments
    model = NMF(n_components=n_components,
                random_state=random_state,
                alpha=alpha,
                l1_ratio=0,
                max_iter=max_iter,
                verbose=verbose,
                solver="cd")

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
                         args.random_state)

            res = run_nmf(arguments)
            w_hat_dict[n_components] = res[0]
            h_hat_dict[n_components] = res[1]
            error_list.append(res[2])

    elif args.nc > 1:
        arguments_list = list()
        for n_components in n_components_list:
            arguments = (data, n_components, args.alpha, args.max_iter, args.verbose, args.random_state)
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

    if args.estimate_rank:
        w_hat, h_hat = estimate_rank(data=tf_idf, args=args)

    else:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
              f"running NMF...")
        arguments = (data, args.n_components, args.alpha,
                     args.max_iter, args.verbose,
                     args.random_state)

        res = run_nmf(arguments)
        w_hat, h_hat = res[0], res[1]

    save_data(w=w_hat, h=h_hat, peaks=peaks, barcodes=barcodes, args=args)

    secs = time.time() - start
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)

    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"total time: {h: .2f}h {m: .2f}m {s: .2f}s")


if __name__ == '__main__':
    main()
