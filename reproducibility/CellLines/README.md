1. RawData ---> process raw sequencing data to generate BAM file for each single cell
2. MergeBAM ---> merge the BAM files and call peaks
3. TagCounts ---> generate peak by cell matrix
4. Imputation ---> obtain imputation matrix for each method
5. DimReduction ---> generate dimension reduction matrix
6. DownstreamAnalysis ---> codes for scABC, Cicero and ChromVAR using either raw or scOpen imputed matrix as input
7. EvaluationByAUPR ---> evaluate the imputation by using AUPR metric
8. ClusteringImputation ---> evaluate the imputation by clustering
9. SilhouetterScoreImputation ---> compute silhouette scores for imputation matrix
10. Visualization ---> visualize imputation using UMAP