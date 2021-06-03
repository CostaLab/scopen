# scOpen: chromatin-accessibility estimation for single-cell ATAC data

## Overview
Single cell ATAC-seq is sparse and high-dimensional, we here propose scOpen to impute
and quantify the open chromatin status of regulatory regions from scATAC-seq data. Moreover,
scOpen provides a low-dimensional matrix for clustering and visualisation.
We show that scOpen improves crucial down-stream analysis steps of scATAC-seq data as clustering, visualisation, 
cis-regulatory DNA interactions and delineation of regulatory features. We demonstrate the power of scOpen to dissect regulatory 
changes in the development of fibrosis in the kidney.

See our [manuscript](https://www.biorxiv.org/content/10.1101/865931v3) for more details.

## Installation
`scopen` has been test with Python 3.6, 3.7, 3.8 and 3.9.  

### Dependencies
[numpy](https://numpy.org/)  
[scipy](https://www.scipy.org/)  
[h5py](https://www.h5py.org/) 
[pandas](https://pandas.pydata.org/)  
[PyTables](http://www.pytables.org/) 
[matplotlib](https://matplotlib.org/)  
[scikit-learn](https://scikit-learn.org/stable/)   
[kneed](https://github.com/arvkevi/kneed)

### User installation
The easiest way to install scopen and the required packages is using `pip`
```commandline
pip install scopen
```

To upgrade to a newer release use the `--upgrade` option:
```commandline
pip install --upgrade scopen
```

Or you can install it from github, first clone the repository:
```commandline
git clone https://github.com/CostaLab/scopen.git
```

and install manually
```commandline
cd scopen
pip install ./
```

## Usage
We here describe how to run `scopen`  

### Input data
`scopen` performs imputation and dimensionality reduction based on peak by 
cell matrix and it allows different input formats. The simplest one is a 
text file where each row represent a peak, and
each column is a cell. For example, you can download 
[here](https://www.dropbox.com/s/pp45n1pcbldeqlq/TagCount.txt.gz?dl=0)
the count matrix from human hematopoietic cells, and uncompress the file:
```commandline
gzip -d TagCount.txt.gz
scopen --input TagCount.txt --input_format dense --output_dir ./ --output_prefix scOpen --output_format dense --verbose 0 --estimate_rank --nc 4
```
We also support different input format, such as scATAC-seq from 10X Genomics.

### Run scopen



Check more information by:
```commandline
scopen --help
```

## Outputs
After the command is done, you can find 5 output files in current directory:
* `scOpen.txt`. An imputed matrix. It has same dimensions as input and can be 
used for downstream analysis, such as peak-to-peak co-accessibility prediction.

* `scOpen_barcodes.txt`. A low-dimension matrix for cells. The number of dimensions is determined by option `--estimate_rank`. 
It can be used as a dimension reduced  matrix for clustering and visualization.

* `scOpen_peaks.txt`. A low-dimension matrix for peaks.

* `scOpen_error.pdf`. A line plot showing the model selection process, where x-axis represent ranks (or dimensions), 
y-axis is the fitting error of NMF. scOpen selects the best model by identifying a elbow point from this curve.

* `scOpen_error.txt`. A text file including data for above curve.

## How to use scOpen in R
scOpen is implemented in python, while many popular tools for analysis scATAC-seq, such as 
[Signac](https://satijalab.org/signac/), [chromVAR](https://github.com/GreenleafLab/chromVAR), are developed using R.
If you are dedicated to R, [here](https://github.com/CostaLab/scopen/blob/master/vignettes/signac_pbmc.Rmd) is 
an example where we used scOpen as a dimension reduction method to analyze a scATAC-seq from human
peripheral blood mononuclear cells (PBMCs) dataset.
