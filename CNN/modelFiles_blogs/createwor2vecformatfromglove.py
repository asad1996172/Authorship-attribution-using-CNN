import pandas as pd
import csv

vector_file = 'glove.840B.300d.txt'

words = pd.read_table(vector_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

print(words.loc['king'].as_matrix())