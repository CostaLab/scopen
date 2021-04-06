1. TagCounts ---> generate peak by cell matrix from Seurat object
2. Imputation ---> obtain imputation matrix for each method
3. DimReduction ---> generate dimension reduction matrix
4. DownstreamAnalysis ---> codes for scABC, Cicero and ChromVAR using either raw or scOpen imputed matrix as input
5. EvaluationByAUPR ---> evaluate the imputation by using AUPR metric
6. ClusteringImputation ---> evaluate the imputation by clustering
7. SilhouetteScoreImputation ---> compute silhouette scores for imputation matrix
8. Visualization ---> visualize imputation using UMAP
9. SilhouetteScoreDimReduction ---> compute silhouette scores for dimension reduced matrix
10. ClusteringDimReduction ---> evaluate the dimension reduced matrix using clustering
11. SilhouetteScoreDownstreamAnalysis ---> compute silhouette scores for downstream analysis
12. VisualizationDownstreamAnalysis ---> visualize the downstream analysis results using UMAP