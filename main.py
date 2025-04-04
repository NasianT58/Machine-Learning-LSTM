#For Numerical Operations and Data Manipulations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#For building and training deep learning models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#1. Load and Preprocess the Data 
#References:
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html

data = pd.read_csv("internet_usage.csv")

years_cols = [str(years) for years in range(2000, 2024)]

for col in years_cols:
    data[col] = pd.to_numeric(data[col], errors = "coerce")

#for col in years_cols:
  #  print(data[col].unique())

#print(data[years_cols].mean())

print(data[years_cols].isna().sum())

data[years_cols].fillna(data[years_cols].mean(), inplace=True)

print(data[years_cols].isna().sum())



#2. Feature Selection and Preparation

#3. Normalize the Data

#4 Create Sequences for the LSTM

#5 Split the Data into Training and Testing Sets

#6. Build the LSTM Model

#7 Evaluate the Model
