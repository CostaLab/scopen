# scOpen: chromatin-accessibility estimation for single-cell ATAC data

## Installation
First some dependencies
```commandline
pip install cython numpy scipy
```

Then install scopen with all other dependencies
```commandline
pip install scopen
```

```commandline
git clone https://github.com/CostaLab/scopen.git
```

and install manually
```commandline
cd scopen
pip install ./
```

## Usage
Download [here](https://www.dropbox.com/s/pp45n1pcbldeqlq/TagCount.txt.gz?dl=0) the count matrix from human hematopoietic cell and run scOpen:
```commandline
gzip -d TagCount.txt.gz
scopen --input TagCount.txt --input_format dense --output_dir ./ --output_prefix scOpen --output_format dense --verbose 0 --estimate_rank --nc 4
```
This matrix contains raw ATAC-seq reads number with each row representing a peak and each column a cell. 
We also support different input format, such as scATAC-seq from 10X Genomics.

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
