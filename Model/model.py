import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Load dataset
df = pd.read_excel("../Dataset/Data_Trained.xlsx")
df.dropna(inplace=True)
# Convert Total_Stops into numeric
df['Total_Stops'] = df['Total_Stops'].map({
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
})
# Date and Time feature engineering
df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
df.drop(["Date_of_Journey"], axis=1, inplace=True)

df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format='mixed').dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"], format='mixed').dt.minute
df.drop(["Dep_Time"], axis=1, inplace=True)

df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"], format='mixed').dt.hour
df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"], format='mixed').dt.minute
df.drop(["Arrival_Time"], axis=1, inplace=True)

# Normalize duration feature
def process_duration(x):
    x = x.strip().lower().replace('h', 'h ').replace('m', 'm ')
    parts = x.strip().split()
    hr, minute = 0, 0
    for part in parts:
        if 'h' in part:
            hr = int(part.replace('h', ''))
        elif 'm' in part:
            minute = int(part.replace('m', ''))
    return hr, minute

df["Duration_hr"], df["Duration_min"] = zip(*df["Duration"].map(process_duration))
df.drop(["Duration"], axis=1, inplace=True)

df.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# Categorical encoding
airline = pd.get_dummies(df["Airline"], drop_first=True)
source = pd.get_dummies(df["Source"], drop_first=True)
destination = pd.get_dummies(df["Destination"], drop_first=True)

df.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)
df = pd.concat([df, airline, source, destination], axis=1)

# Train-test split 
X = df.drop("Price", axis=1)
y = df["Price"]

# Train the model
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model
with open("flight_rf.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model retrained and saved to 'flight_rf.pkl'")
