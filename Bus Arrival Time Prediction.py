#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[3]:


dat=pd.read_csv("Arrival Data.csv")
dat


# In[4]:


dat.isnull().sum()


# In[6]:


#dat.corr()


# In[10]:


#Data preprocessing
#converting day to numeric
Day_mapping={
    'Mon':1,
    'Tues':2,
    'Wed':3,
    'Thur':4,
    'Fri':5,
    'Sat':6,
    
}


# In[11]:


dat['Day_encoded']=dat['Day'].map(Day_mapping)


# In[13]:


#Traffic level conversion
Traffic_level_mapping={
    'high':10,
    'low':3,
    'medium':6,
}


# In[14]:


dat['Traffic_level_encoded']=dat['Traffic level'].map(Traffic_level_mapping)


# In[8]:


df=pd.read_csv("T_Data.csv")
df


# In[9]:


df.corr()


# In[17]:


#Exploratory Data Analysis
# Time Series Analysis
fig_time_series = go.Figure()
fig_time_series.add_trace(go.Scatter(x=df['Day'], y=df['Arrival_time'], mode='lines+markers', name='Arrival Time'))
fig_time_series.add_trace(go.Scatter(x=df['Day'], y=df['Departure'], mode='lines+markers', name='Departure Time'))
fig_time_series.update_layout(title='Bus Arrival and Departure Times Weekly Analysis',
                              xaxis_title='Day',
                              yaxis_title='Time')
fig_time_series.show()


# In[18]:


# Day of the Week Analysis
fig_day_of_week = go.Figure(data=[go.Pie(labels=df['Day'], values=df['Day'])])
fig_day_of_week.update_layout(title='Distribution of Bus Records by Day of the Week')
fig_day_of_week.show()


# In[19]:


# Traffic Level Analysis
fig_traffic_level = go.Figure(data=[go.Bar(x=df['Day'], y=df['Traffic_level'])])
fig_traffic_level.update_layout(title='Traffic Level Distribution',
                                xaxis_title='Day',
                                yaxis_title='Traffic Level')
fig_traffic_level.show()


# In[20]:


# Bus Analysis
fig_bus = go.Figure(data=[go.Bar(x=df['Day'], y=df['Bus_ID'])])
fig_bus.update_layout(title='Distribution of Bus Records by Bus ID',
                      xaxis_title='Day',
                      yaxis_title='Bus ID')
fig_bus.show()


# In[74]:


#Training the model
# Split the data into features (X) and target variables (y)
X = df[['Time', 'Day', 'Traffic_level', 'Route_ID', 'Bus_ID']]
y_arrival = df['Arrival_time']
y_departure = df['Departure']


# In[75]:


# Split the data into training and testing sets
X_train, X_test, y_arrival_train, y_arrival_test = train_test_split(X, y_arrival, test_size=0.2, random_state=42)
X_train, X_test, y_departure_train, y_departure_test = train_test_split(X, y_departure, test_size=0.2, random_state=42)


# In[76]:


# Train the Random Forest model for arrival time prediction
rf_arrival = RandomForestRegressor(random_state=42)
rf_arrival.fit(X_train, y_arrival_train)


# In[77]:


# Train the Random Forest model for departure time prediction
rf_departure = RandomForestRegressor(random_state=42)
rf_departure.fit(X_train, y_departure_train)


# In[78]:


# Make predictions for arrival time
y_arrival_pred = rf_arrival.predict(X_test)

# Make predictions for departure time
y_departure_pred = rf_departure.predict(X_test)


# In[79]:


# User input for new prediction
new_data = {
    'Time': [740],
    'Day': [2],
    'Traffic_level': [3],
    'Route_ID': [3],
    'Bus_ID': [6]
}

new_df = pd.DataFrame(new_data)

# Make predictions for arrival and departure times
new_arrival_pred = rf_arrival.predict(new_df)
new_departure_pred = rf_departure.predict(new_df)

#print("Predicted Arrival Time:", new_arrival_pred[0])
#print("Predicted Departure Time:", new_departure_pred[0])


# In[80]:


# Convert decimal predictions to time objects
arrival_time = datetime.strptime("0800", "%H%M") + timedelta(minutes=new_arrival_pred[0])
departure_time = datetime.strptime("0800", "%H%M") + timedelta(minutes=new_departure_pred[0])

# Format the time objects as strings
formatted_arrival_time = arrival_time.strftime("%H:%M")
formatted_departure_time = departure_time.strftime("%H:%M")


# In[81]:


print("Predicted Arrival Time:", formatted_arrival_time)
print("Predicted Departure Time:", formatted_departure_time)


# In[83]:


#Performance Metrics analysis
# Split the data into features (X) and target variables (y)
X = df[['Time', 'Day', 'Traffic_level', 'Route_ID', 'Bus_ID']]
y_arrival = df['Arrival_time']
y_departure = df['Departure']


# In[84]:


# Train the Random Forest model for arrival time prediction
rf_arrival = RandomForestRegressor(random_state=42)
rf_arrival.fit(X, y_arrival)


# In[85]:


# Train the Random Forest model for departure time prediction
rf_departure = RandomForestRegressor(random_state=42)
rf_departure.fit(X, y_departure)


# In[86]:


# Make predictions for arrival and departure times
X_new = pd.DataFrame({
    'Time': [915],
    'Day': [7],
    'Traffic_level': [8],
    'Route_ID': [2],
    'Bus_ID': [6]
})


# In[87]:


# Make predictions for arrival and departure times
new_arrival_pred = rf_arrival.predict(X_new)
new_departure_pred = rf_departure.predict(X_new)


# In[88]:


# Calculate evaluation metrics
#mse_arrival = mean_squared_error(y_arrival, rf_arrival.predict(X))
#mse_departure = mean_squared_error(y_departure, rf_departure.predict(X))


# In[89]:


r2_arrival = r2_score(y_arrival, rf_arrival.predict(X))
r2_departure = r2_score(y_departure, rf_departure.predict(X))


# In[90]:


mae_arrival = mean_absolute_error(y_arrival, rf_arrival.predict(X))
mae_departure = mean_absolute_error(y_departure, rf_departure.predict(X))


# In[92]:


#print("Mean Squared Error (Arrival Time):", mse_arrival)
#print("Mean Squared Error (Departure Time):", mse_departure)
print("R-squared (Arrival Time):", r2_arrival)
print("R-squared (Departure Time):", r2_departure)
print("Mean Absolute Error (Arrival Time):", mae_arrival)
print("Mean Absolute Error (Departure Time):", mae_departure)


# In[ ]:




