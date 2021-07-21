# scOpen: chromatin-accessibility estimation for single-cell ATAC data
Current version for `scopen` is 0.1.7

## System Requirements
`scopen` has been test with following OS:  
macOS Big Sur (11.4)  
Linux (4.18.0)

## Installation
`scopen` has been test with Python 3.6, 3.7, 3.8 and 3.9.
We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to setup
the environment.

### Dependencies
[numpy](https://numpy.org/) (>=1.20.3)  
[scipy](https://www.scipy.org/) (>=1.6.3)  
[h5py](https://www.h5py.org/) (>=3.2.1)  
[pandas](https://pandas.pydata.org/) (>=1.2.4)  
[PyTables](http://www.pytables.org/) (>=3.6.1)  
[matplotlib](https://matplotlib.org/) (>=3.4.2)   
[scikit-learn](https://scikit-learn.org/stable/) (>=0.24.2)     
[kneed](https://github.com/arvkevi/kneed) (>=0.7.0)

### User installation
The easiest way to install scopen and the required packages is using `pip`
```commandline
pip install ./
```

The installation will take ~20 seconds.

## Usage
We here describe how to run `scopen`.

### Input data
`scopen` performs imputation and dimensionality reduction based on peak by 
cell matrix and it allows different input formats. The simplest one is a 
text file where each row represent a peak, and
each column is a cell. 

Here, we provide an example data in `demo` folder, which is a
peak by cell count matrix from human hematopoietic cells. 

First uncompress the file:
```commandline
cd demo
gzip -d TagCount.txt.gz
```

### Run scopen
Execute below command to run scopen:
```commandline
scopen --input TagCount.txt --input_format dense --output_dir ./ --output_prefix scOpen --output_format dense --verbose 0 --estimate_rank --nc 4
```
`--input_format`: this option specifies the input format as dense for which
a text file is expected    
`--output_dir`: all output files will be saved in current directory  
`--output_prefix`: output file name  
`--verbose`: verbose level  
`--estimate_rank`: the number of ranks will be automatically selected  
`--nc`: how many cores will be used

See more information by:
```commandline
scopen --help
```

The expected running time is ~18 minutes.

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

## How to run scOpen for your data
As described about, `scopen` also supports following input formats.
* 10X: a folder including barcodes.tsv, matrix.mtx and peaks.bed, 
it is usually generated by using cellranger-atac pipeline.
* 10Xh5: a peak barcode matrix in hdf5 format. 
* sparse: a text file including three columns,
the first column indicates peaks, the secondcolumns
represent barcodes and the third one is the number of reads

## How to use scOpen in R
scOpen is implemented in python, while many popular tools for analysis scATAC-seq, such as 
[Signac](https://satijalab.org/signac/), are developed using R.
If you are dedicated to R, we also provide a tutorial 
[here](https://github.com/CostaLab/scopen/blob/master/vignettes/signac_pbmc.Rmd) to 
show you how use `scopen` as a dimension reduction method in R to analyze scATAC-seq data 
from human peripheral blood mononuclear cells (PBMCs) dataset.

## How to combine scOpen and (epi)scanpy
Python is gaining popularity in single-cell data analysis. 
Two examples are scanpy (for scRNA-seq) and episcanpy (for single cell epigenomic data, e.g., scATAC-seq).
To ensure `scopen` is usable in this context, we provide a [jupyter notebook](https://github.com/CostaLab/scopen/blob/master/vignettes/epiScanpy.ipynb) to
show you how to combine scOpen and (epi)scanpy to analysis scATAC-seq data.

## Reproduction
For reproducibility, we provide all scripts and data [here](https://github.com/CostaLab/scopen-reproducibility).