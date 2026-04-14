import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # 필요한 전처리
    df = df.dropna()
    return df
