import os
import pandas as pd
import numpy as np
import scipy.sparse as sp_sparse
from scipy.io.mmio import mmread, mmwrite
import tables
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def load_data(args):
    if args.input_format == "sparse":
        data, barcodes, peaks = get_data_from_sparse_file(filename=args.input)

    elif args.input_format == "dense":
        data, barcodes, peaks = get_data_from_dense_file(filename=args.input)

    elif args.input_format == "10Xh5":
        data, barcodes, peaks = get_data_from_10x_h5(filename=args.input)

    elif args.input_format == "10X":
        data, barcodes, peaks = get_data_from_10x(input_dir=args.input)

    return data, barcodes, peaks


def save_data(w, h, peaks, barcodes, args):
    m_hat = np.dot(w, h)
    df = pd.DataFrame(data=w, index=peaks)
    df.to_csv(os.path.join(args.output_dir, "{}_peaks.txt".format(args.output_prefix)), sep="\t")

    df = pd.DataFrame(data=h, columns=barcodes)
    df.to_csv(os.path.join(args.output_dir, "{}_barcodes.txt".format(args.output_prefix)), sep="\t")

    if args.output_format == "sparse":
        filename = os.path.join(args.output_dir, "{}.txt".format(args.output_prefix))
        write_data_to_sparse_file(filename=filename, data=m_hat, barcodes=barcodes, peaks=peaks)

    elif args.output_format == "dense":
        filename = os.path.join(args.output_dir, "{}.txt".format(args.output_prefix))
        write_data_to_dense_file(filename=filename, data=m_hat, barcodes=barcodes, peaks=peaks)

    elif args.output_format == "10X":
        write_data_to_10x(output_dir=args.output_dir, data=m_hat, barcodes=barcodes, peaks=peaks)


def get_data_from_sparse_file(filename):
    peaks = []
    barcodes = []
    with open(filename) as f:
        for line in f:
            ll = line.strip().split("\t")
            peaks.append(ll[0])
            barcodes.append(ll[1])

    peaks = list(set(peaks))
    barcodes = list(set(barcodes))

    df = pd.DataFrame(0, columns=barcodes, index=peaks)

    with open(filename) as f:
        for line in f:
            ll = line.strip().split("\t")
            peak = ll[0]
            cell = ll[1]
            df.at[peak, cell] = int(ll[2])

    data = df.values.astype(np.float16)
    barcodes = df.columns.values.tolist()
    peaks = df.index.values

    del df

    return data, barcodes, peaks


def get_data_from_dense_file(filename):
    df = pd.read_csv(filename, delimiter="\t", index_col=0)

    data = df.values.astype(np.float16)
    barcodes = df.columns.values.tolist()
    peaks = df.index.values

    del df

    return data, barcodes, peaks


def get_data_from_10x(input_dir):
    matrix_file = os.path.join(input_dir, "matrix.mtx")
    barcodes_file = os.path.join(input_dir, "barcodes.tsv")
    peaks_file = os.path.join(input_dir, "peaks.bed")

    data = mmread(matrix_file).todense(order='F')
    barcodes = list()
    peaks = list()

    with open(barcodes_file) as f:
        for line in f:
            ll = line.strip().split("\t")
            barcodes.append(ll[0])

    with open(peaks_file) as f:
        for line in f:
            ll = line.strip().split("\t")
            peaks.append(ll[0] + ":" + ll[1] + "-" + ll[2])

    return np.asarray(data, dtype=np.float16), barcodes, peaks


def get_data_from_10x_h5(filename):
    with tables.open_file(filename, 'r') as f:
        try:
            group = f.get_node(f.root, 'matrix')
        except tables.NoSuchNodeError:
            print("Matrix group does not exist in this file.")
            return None

        feature_group = getattr(group, 'features').read()
        ids = getattr(feature_group, 'id').read()
        names = getattr(feature_group, 'name').read()
        barcodes = getattr(group, 'barcodes').read()
        data = getattr(group, 'data').read()
        indices = getattr(group, 'indices').read()
        indptr = getattr(group, 'indptr').read()
        shape = getattr(group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape).todense(order='F')

        return np.asarray(matrix, dtype=np.float16), barcodes, names


def write_data_to_sparse_file(filename, data, barcodes, peaks):
    df = pd.DataFrame(data=data, columns=barcodes, index=peaks)
    with open(filename, "w") as f:
        for peak in df.index.values.tolist():
            for barcode in df.columns.values.tolist():
                if df.at[peak, barcode] > 0:
                    f.write("\t".join(map(str, [peak, barcode, df.at[peak, barcode]])) + "\n")


def write_data_to_dense_file(filename, data, barcodes, peaks):
    df = pd.DataFrame(data=data, columns=barcodes, index=peaks)
    df.to_csv(filename, sep="\t")


def write_data_to_10x(output_dir, data, barcodes, peaks):
    matrix_file = os.path.join(output_dir, "matrix.mtx")
    barcodes_file = os.path.join(output_dir, "barcodes.tsv")
    peaks_file = os.path.join(output_dir, "peaks.bed")

    with open(matrix_file, "w") as f:
        # write initial header line
        f.write('%%MatrixMarket matrix coordinate real general\n')
        (rows, cols) = data.shape

        # write shape spec
        f.write('{} {} {}\n'.format(rows, cols, np.count_nonzero(data)))

        for col in range(cols):
            for row in range(rows):
                if data[row, col] > 0:
                    f.write("{} {} {}\n".format(row + 1, col + 1, data[row, col]))

    with open(barcodes_file, "w") as f:
        for barcode in barcodes:
            f.write(barcode + "\n")

    with open(peaks_file, "w") as f:
        for peak in peaks:
            chrom = peak.split(":")[0]
            start = peak.split(":")[-1].split("-")[0]
            end = peak.split(":")[-1].split("-")[1]

            f.write("\t".join([chrom, start, end]) + "\n")


def plot_open_regions_density(n_open_regions, args):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(np.log10(n_open_regions), density=True, bins=100)

    ax.set_xlabel("Number of detected peaks (log10)")
    ax.set_ylabel("Density")

    output_filename = os.path.join(args.output_dir, "{}_peaks.pdf".format(args.output_prefix))
    fig.tight_layout()
    fig.suptitle('Number of detected peaks per cell')
    fig.savefig(output_filename)


def plot_estimated_dropout(rho, args):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(rho, density=True, bins=100)

    ax.set_xlabel("Estimated dropout rate")
    ax.set_ylabel("Density")

    output_filename = os.path.join(args.output_dir, "{}_estimated_dropout.pdf".format(args.output_prefix))
    fig.tight_layout()
    fig.suptitle('Estimated dropout rate per cell')
    fig.savefig(output_filename)


def plot_objective(obj, args):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(obj)

    ax.set_xlabel("Iter")
    ax.set_ylabel("Loss")

    output_filename = os.path.join(args.output_dir, "{}_loss.pdf".format(args.output_prefix))
    fig.tight_layout()
    fig.savefig(output_filename)


def plot_error(n_components_list, train_error_list, test_error_list, best_rank, args):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(n_components_list, train_error_list, label="train")
    ax2.plot(n_components_list, test_error_list, label="test")

    ax1.axvline(x=best_rank)
    ax2.axvline(x=best_rank)

    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Error")
    ax1.set_title("Training")

    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Error")
    ax2.set_title("Test")

    output_filename = os.path.join(args.output_dir, "{}_error.pdf".format(args.output_prefix))
    fig.tight_layout()
    fig.set_size_inches(w=8, h=4)
    fig.savefig(output_filename)

    output_filename = os.path.join(args.output_dir, "{}_error.txt".format(args.output_prefix))
    df = pd.DataFrame({'ranks': n_components_list,
                       'training_error': train_error_list,
                       'test_error': test_error_list})

    df.to_csv(output_filename, index=False, sep="\t")
