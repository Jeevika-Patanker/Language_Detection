import pandas as pd

def load_dataset(path):
    dataframe = pd.read_csv(path) 
    data = dataframe[['text','labels']]
    return data
 