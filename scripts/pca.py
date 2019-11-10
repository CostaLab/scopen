import os
import sys
import numpy as np
import pandas as pd
from scopen.utils import get_data_from_dense_file
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

input_file = sys.argv[1]
output_location = sys.argv[2]
output_prefix = sys.argv[3]
n_components = int(sys.argv[4])

data, barcodes, peaks = get_data_from_dense_file(input_file)
scaler = StandardScaler(copy=False)
data = scaler.fit_transform(data)

pca = decomposition.PCA(n_components=n_components)
data = pca.fit_transform(np.transpose(data))

output_filename = os.path.join(output_location, "{}.txt".format(output_prefix))
df = pd.DataFrame(data=np.transpose(data), columns=barcodes)
df.to_csv(output_filename, sep="\t")
