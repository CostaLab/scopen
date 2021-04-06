1. RawData ---> process raw sequencing data to generate BAM file for each single cell
2. MergeBAM ---> merge the BAM files and call peaks
3. TagCounts ---> generate peak by cell matrix
4. Imputation ---> obtain imputation matrix for each method
5. DimReduction ---> generate dimension reduction matrix