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
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler

#Load the Data
data = pd.read_csv("internet_usage.csv")

#get the columns that are years in ranges 2000, 2024
years_cols = [str(years) for years in range(2000, 2024)]

#convert the data in those columns to numeric
for col in years_cols:
    data[col] = pd.to_numeric(data[col], errors = "coerce")

#get the mean of the data and replace the NaN values with them
data[years_cols] = data[years_cols].fillna(data[years_cols].mean())

#2. Feature Selection and Preparation
data_filtered = data.filter(items = years_cols)

#3. Normalize the Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_filtered_scaled = scaler.fit_transform(data_filtered)

#4 Create Sequences for the LSTM

data_sequenced = np.array(data_filtered_scaled)


#5 Split the Data into Training and Testing Sets

#6. Build the LSTM Model

#7 Evaluate the Model
