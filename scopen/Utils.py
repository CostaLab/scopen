import os
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.sparse as sp_sparse
from scipy.io.mmio import mmread
import tables
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


def load_data(args):
    print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}, "
          f"loading data...")

    if args.input_format == "sparse":
        data, barcodes, peaks = get_data_from_sparse_file(filename=args.input)

    elif args.input_format == "dense":
        data, barcodes, peaks = get_data_from_dense_file(filename=args.input)

    elif args.input_format == "10Xh5":
        data, barcodes, peaks = get_data_from_10x_h5(filename=args.input)

    elif args.input_format == "10X":
        data, barcodes, peaks = get_data_from_10x(input_dir=args.input)

    return data, barcodes, peaks


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

    data = df.values.astype(np.int8)
    data = sp_sparse.csr_matrix(data)

    barcodes = df.columns.values.tolist()
    peaks = df.index.values

    del df

    return data, barcodes, peaks


def get_data_from_dense_file(filename):
    df = pd.read_csv(filename, delimiter="\t", index_col=0)

    data = df.values.astype(np.int8)
    data = sp_sparse.csr_matrix(data)

    barcodes = df.columns.values.tolist()
    peaks = df.index.values

    del df

    return data, barcodes, peaks


def get_data_from_10x(input_dir):
    matrix_file = os.path.join(input_dir, "matrix.mtx")
    barcodes_file = os.path.join(input_dir, "barcodes.tsv")
    peaks_file = os.path.join(input_dir, "peaks.bed")

    # use sparse matrix to save memory
    data = mmread(matrix_file).astype(dtype=np.int8)
    data = sp_sparse.csr_matrix(data)
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

    return data, barcodes, peaks


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
        matrix = sp_sparse.csr_matrix((data, indices, indptr), shape=shape)

        return matrix, barcodes, names


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


def plot_open_regions_density(data, args):
    n_open_regions = np.count_nonzero(data, axis=0)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(np.log10(n_open_regions), density=True, bins=100)

    ax.set_xlabel("Number of detected peaks (log10)")
    ax.set_ylabel("Density")

    output_filename = os.path.join(args.output_dir, "{}_peaks.pdf".format(args.output_prefix))
    fig.tight_layout()
    fig.suptitle('Number of detected peaks per cell')
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


def plot_knee(kl, args):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.plot(kl.x, kl.y, "bx-")
    ax.axvline(kl.knee, linestyle="--", label="knee")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Error")
    ax.set_title(f"Knee Point: {kl.knee}")

    output_filename = os.path.join(args.output_dir, "{}_error.pdf".format(args.output_prefix))
    fig.tight_layout()
    fig.savefig(output_filename)

    output_filename = os.path.join(args.output_dir, "{}_error.txt".format(args.output_prefix))
    df = pd.DataFrame({'ranks': kl.x,
                       'error': kl.y})

    df.to_csv(output_filename, index=False, sep="\t")
