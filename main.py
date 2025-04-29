import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("Dataset/3997445.csv")
    print(df)
    print(df[['TAVG', 'PRCP', 'SNOW']])
    test = df.dropna(subset=['TAVG', 'PRCP', 'SNOW'])
    print(test[['TAVG', 'PRCP', 'SNOW']])

if __name__ == '__main__':
    main()