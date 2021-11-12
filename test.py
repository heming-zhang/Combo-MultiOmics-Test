import numpy as np
import pandas as pd

# a = np.load('./datainfo/form_data/x_split2.npy')
# print(a.shape)

# print(a[0,0:45])

final_drugbank_df = pd.read_csv('./datainfo/filtered_data/final_drugbank.csv')
gene_list = list(set(final_drugbank_df['Target']))
print(len(gene_list))
