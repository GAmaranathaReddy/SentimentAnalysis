import pandas as pd

def visualization():
    print("Function is called")
    df=pd.read_csv("../data/data_distinct.csv")
    #print(df.head())
    #groupcount=df.groupby('landscapeid')
    #print(df['landscapeid'].value_counts())
    print(df['landscapeid'].hist())


visualization()