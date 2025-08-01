import pandas as pd
import numpy as np

df = pd.read_csv("CustomersDataSet")

df[:5]

df.info()
df.isnull().sum()


df['Profession'].fillna('Unknown', inplace=True)

def segment(row):
    if row['Annual Income ($)'] > 70000 and row['Spending Score (1-100)'] > 70:
        return 'High Income / High Spending'
    elif row['Annual Income ($)'] > 70000:
        return 'High Income / Low Spending'
    elif row['Spending Score (1-100)'] > 70:
        return 'Low Income / High Spending'
    else:
        return 'Others'

df['Segment'] = df.apply(segment, axis=1)

df[:5]

df['Income per Capita'] = df['Annual Income ($)'] / df['Family Size'].replace(0, np.nan)
df[:5]


