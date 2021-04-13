# scOpen: chromatin-accessibility estimation of single-cell ATAC data

## Installation
First some dependencies \
`$ pip install --user cython numpy scipy`

Then install scopen with all other dependencies \
`$ pip install scopen`

Alternatively (but not recommended), you can clone this repository: \
`$ git clone git@github.com:CostaLab/scopen.git`

and install manually \
`$ cd scopen` \
`$ python setup.py install`

## Usage
Download [here](https://costalab.ukaachen.de/open_data/scOpen/HematopoieticCells.txt) the count matrix from human 
hematopoietic cell and run scOpen: \
`$ scopen --input HematopoieticCells.txt --input-format dense --output-dir ./ --output-prefix HematopoieticCells_scOpen --output-format dense`

This matrix contains raw ATAC-seq reads number with each row representing a peak and each column a cell. 
We also support different input format, such as scATAC-seq from 10X Genomics.

Check more information by: \
`$ scopen --help`


## Outputs
After the command is done, three files are generated in current directory:
* `HematopoieticCells_scOpen.txt`. The estimated matrix by scOpen. This matrix has same dimensions as input and can be 
used for downstream analysis, such as clustering, visualization.

* `HematopoieticCells_scOpen_barcodes.txt` and `HematopoieticCells_scOpen_peaks.txt` contain low dimensional representation 
for cells and peaks.
  
## How to use scOpen in R
If you want to use scOpen in R, [here](https://github.com/CostaLab/scopen/blob/master/vignettes/signac_pbmc.Rmd) is a tutorial. 
