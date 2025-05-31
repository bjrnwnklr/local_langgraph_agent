import pandas as pd
import csv

df = pd.read_csv("test_data_set.csv")

df.to_csv("test_data_set_quoted.csv", index=False, quotechar='"', quoting=csv.QUOTE_ALL)
