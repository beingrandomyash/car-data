# car-data
#car data data science project 
import numpy as np
import pandas as pd
from sklearn import linear_model




from google.colab import files
uploaded = files.upload()
df= pd.read_csv("car data.csv")
df
inputs =df.drop(['Car_Name', 'Owner', 'Selling_type'], axis='columns')
target=df.Selling_type
inputs
from sklearn.preprocessing import LabelEncoder
numerics= LabelEncoder()

inputs['Fuel_Type']=numerics.fit_transform(inputs['Fuel_Type'])

inputs['Transmission']=numerics.fit_transform(inputs['Transmission'])
inputs
inputs_n=inputs.drop(['Fuel_Type', 'Transmission'], axis='columns')
inputs_n
model= linear_model.LinearRegression()

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
model = LinearRegression()

# Fit the model with inputs_n and target_encoded
model.fit(inputs_n, target_encoded)
pred=model.predict([[2013,430000, 1, 1]])
print (pred)
df
import matplotlib.pyplot as plt

training_pred=model.predict(inputs_n)

from sklearn import metrics
r2 = r2_score(target_encoded, predictions)
print('R-squared Score:', r2)
plt.scatter(target_encoded, training_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
