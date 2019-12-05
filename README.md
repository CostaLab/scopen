# scOpen: estimate single cell open chromatin accessibility

# Installation
First some dependencies \
`pip install --user cython numpy scipy`

Then install scopen with all other dependencies \
`pip install scopen`

Alternatively (but not recommended), you can clone this repository: \
`git clone git@github.com:CostaLab/scopen.git`

and install manually \
`cd scopen` \
`python setup install`

# Usage
Download [here](https://costalab.ukaachen.de/open_data/scOpen/HematopoieticCells.txt) the count matrix from human 
hematopoietic cell and run scOpen: \
`scopen --input HematopoieticCells.txt --input-format dense --output-dir ./ --output-prefix HematopoieticCells_scOpen --output-format dense`
