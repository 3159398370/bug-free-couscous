#!/usr/bin/env python
# coding: utf-8

# In[85]:


import sys


# In[86]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[87]:


# Display settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


# In[88]:


# Load datasets
input_train_filepath = sys.argv[1]
traffic_df = pd.read_csv(input_train_filepath)
traffic_two_month_df = pd.read_csv(input_train_filepath)


# In[89]:


traffic_df.head(10)


# In[90]:


traffic_two_month_df.head(10)


# In[91]:


traffic_df['Traffic Situation'].value_counts()


# In[92]:


# Combine datasets
traffic_df['Source'] = 'OneMonth'
traffic_two_month_df['Source'] = 'TwoMonth'
combined_df = pd.concat([traffic_df, traffic_two_month_df], ignore_index=True)


# In[93]:


# 1. Distribution of vehicle counts for cars, bikes, buses, and trucks
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts", "Bike Counts", "Bus Counts", "Truck Counts"))

fig.add_trace(go.Histogram(x=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Histogram(x=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Histogram(x=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Histogram(x=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Distribution of Vehicle Counts', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Count")
fig.update_yaxes(title_text="Frequency")
#fig.show()


# In[94]:


# 3. Distribution of traffic situations
fig = px.pie(combined_df, names='Traffic Situation', title='Traffic Situation Distribution', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Traffic Situation Distribution', title_x=0.5, template='plotly_white')
#fig.show()


# In[95]:


# 4. Vehicle count vary by day of the week
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Day", "Bike Counts by Day", "Bus Counts by Day", "Truck Counts by Day"))

fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Vehicle Counts by Day of the Week', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Day of the Week")
fig.update_yaxes(title_text="Count")
fig.show()


# In[96]:


# 5-8. Relationship between vehicle counts and traffic situation
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Traffic Situation", "Bike Counts by Traffic Situation", "Bus Counts by Traffic Situation", "Truck Counts by Traffic Situation"))

fig.add_trace(go.Box(x=combined_df['Traffic Situation'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Traffic Situation'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Traffic Situation'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Traffic Situation'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Vehicle Counts by Traffic Situation', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Traffic Situation")
fig.update_yaxes(title_text="Count")
#fig.show()


# In[97]:


# 9. Total vehicle count by traffic situation
fig = px.box(combined_df, x='Traffic Situation', y='Total', title='Total Vehicle Count by Traffic Situation', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Total Vehicle Count by Traffic Situation', title_x=0.5, xaxis_title='Traffic Situation', yaxis_title='Total Count', template='plotly_white')
#fig.show()


# In[98]:


# 10. Vehicle count vary by source
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Source", "Bike Counts by Source", "Bus Counts by Source", "Truck Counts by Source"))

fig.add_trace(go.Box(x=combined_df['Source'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Source'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Source'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Source'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Vehicle Counts by Source', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Source")
fig.update_yaxes(title_text="Count")
#fig.show()


# In[99]:


# 11. Busiest hours of the day for traffic
combined_df['Hour'] = pd.to_datetime(combined_df['Time'], format='%I:%M:%S %p').dt.hour
fig = px.box(combined_df, x='Hour', y='Total', title='Total Vehicle Count by Hour', color_discrete_sequence=['#9467bd'])
fig.update_layout(title_text='Total Vehicle Count by Hour', title_x=0.5, xaxis_title='Hour', yaxis_title='Total Count', template='plotly_white')
#fig.show()


# In[100]:


# 12. Traffic situation distribution by day of the week
fig = px.box(combined_df, x='Day of the week', y='Traffic Situation', title='Traffic Situation by Day of the Week', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Traffic Situation by Day of the Week', title_x=0.5, xaxis_title='Day of the Week', yaxis_title='Traffic Situation', template='plotly_white')
#fig.show()


# In[101]:


# 13. Correlations between different vehicle types
corr_matrix = combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Vehicle Counts')
#plt.show()


# In[102]:


# 16. Distribution of vehicle counts between weekdays and weekends
combined_df['Weekend'] = combined_df['Day of the week'].isin(['Saturday', 'Sunday'])
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Weekend", "Bike Counts by Weekend", "Bus Counts by Weekend", "Truck Counts by Weekend"))

fig.add_trace(go.Box(x=combined_df['Weekend'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Weekend'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Weekend'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Weekend'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Vehicle Counts by Weekend', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Weekend")
fig.update_yaxes(title_text="Count")
#fig.show()


# In[103]:


# 18. Distribution of traffic situations by hour
fig = px.box(combined_df, x='Hour', y='Traffic Situation', title='Traffic Situation by Hour', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Traffic Situation by Hour', title_x=0.5, xaxis_title='Hour', yaxis_title='Traffic Situation', template='plotly_white')
#fig.show()


# In[104]:


# Calculate average total vehicle count for each source
combined_df['Traffic Situation'] = combined_df['Traffic Situation'].astype('category').cat.codes
avg_source_counts = combined_df.groupby('Source').mean(numeric_only=True).reset_index()
fig = px.bar(avg_source_counts, x='Source', y='Total', title='Average Total Vehicle Count by Source', color_discrete_sequence=['#1f77b4'])
fig.update_layout(title_text='Average Total Vehicle Count by Source', title_x=0.5, xaxis_title='Source', yaxis_title='Average Total Count', template='plotly_white')
#fig.show()


# In[105]:


# 20. Distribution of total vehicle counts for each day of the week
fig = px.box(combined_df, x='Day of the week', y='Total', title='Total Vehicle Count by Day of the Week', color_discrete_sequence=['#9467bd'])
fig.update_layout(title_text='Total Vehicle Count by Day of the Week', title_x=0.5, xaxis_title='Day of the Week', yaxis_title='Total Count', template='plotly_white')
#fig.show()


# In[106]:


# 22. Variance in vehicle counts across vehicle types
variance_df = combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].var().reset_index()
variance_df.columns = ['Vehicle Type', 'Variance']
fig = px.bar(variance_df, x='Vehicle Type', y='Variance', title='Variance in Vehicle Counts', color_discrete_sequence=['#ff7f0e'])
fig.update_layout(title_text='Variance in Vehicle Counts', title_x=0.5, xaxis_title='Vehicle Type', yaxis_title='Variance', template='plotly_white')
#fig.show()


# In[107]:


# 23. Distribution of vehicle counts by source and day of the week
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Source and Day", "Bike Counts by Source and Day", "Bus Counts by Source and Day", "Truck Counts by Source and Day"))

fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Distribution of Vehicle Counts by Source and Day of the Week', title_x=0.5, showlegend=False, template='plotly_white')
#fig.show()


# In[108]:


# 25. Peak traffic hours for each vehicle type
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Hour", "Bike Counts by Hour", "Bus Counts by Hour", "Truck Counts by Hour"))

fig.add_trace(go.Box(x=combined_df['Hour'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Hour'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Hour'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Hour'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

fig.update_layout(title_text='Vehicle Counts by Hour', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Hour")
fig.update_yaxes(title_text="Count")
#fig.show()


# In[109]:


# 26. Average vehicle count for each type change over time
# Exclude non-numeric columns from the mean calculation
numeric_cols = combined_df.select_dtypes(include=['number']).columns

# Group by 'Time' and calculate the mean for numeric columns
avg_time_counts = combined_df.groupby('Time')[numeric_cols].mean().reset_index()

# Create the line plot
fig = px.line(avg_time_counts, x='Time', y=['CarCount', 'BikeCount', 'BusCount', 'TruckCount'], title='Average Vehicle Counts Over Time')
fig.update_layout(title_text='Average Vehicle Counts Over Time', title_x=0.5, xaxis_title='Time', yaxis_title='Average Count', template='plotly_white')

# Display the plot
fig.show()


# In[110]:


# 29. Distribution of traffic situations by date
fig = px.box(combined_df, x='Date', y='Traffic Situation', title='Traffic Situation by Date', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_layout(title_text='Traffic Situation by Date', title_x=0.5, xaxis_title='Date', yaxis_title='Traffic Situation', template='plotly_white')
fig.show()


# In[111]:


# Identify and remove outliers
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

vehicle_counts = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
combined_df = remove_outliers(combined_df, vehicle_counts)


# In[112]:


# Check for missing values and duplicates
print("Missing values in each column:")
print(combined_df.isnull().sum())

print(f"Number of duplicate rows: {combined_df.duplicated().sum()}")


# In[113]:


# Plot boxplots to visualize outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.boxplot(data=combined_df, x='CarCount', ax=axes[0, 0], color='#1f77b4')
sns.boxplot(data=combined_df, x='BikeCount', ax=axes[0, 1], color='#ff7f0e')
sns.boxplot(data=combined_df, x='BusCount', ax=axes[1, 0], color='#2ca02c')
sns.boxplot(data=combined_df, x='TruckCount', ax=axes[1, 1], color='#d62728')
axes[0, 0].set_title('Car Count')
axes[0, 1].set_title('Bike Count')
axes[1, 0].set_title('Bus Count')
axes[1, 1].set_title('Truck Count')
plt.tight_layout()
plt.show()


# In[114]:


# Check normality for each vehicle count before normalization
def check_normality(data):
    stat, p = shapiro(data)
    return p > 0.05

print("Normality check before normalization:")
car_normal = check_normality(combined_df['CarCount'])
bike_normal = check_normality(combined_df['BikeCount'])
bus_normal = check_normality(combined_df['BusCount'])
truck_normal = check_normality(combined_df['TruckCount'])

print(f"Car count normality: {car_normal}")
print(f"Bike count normality: {bike_normal}")
print(f"Bus count normality: {bus_normal}")
print(f"Truck count normality: {truck_normal}")

# Normalize data using QuantileTransformer
scaler = QuantileTransformer(output_distribution='normal')
combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = scaler.fit_transform(combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])

# Check normality for each vehicle count after normalization
print("Normality check after normalization:")
car_normal = check_normality(combined_df['CarCount'])
bike_normal = check_normality(combined_df['BikeCount'])
bus_normal = check_normality(combined_df['BusCount'])
truck_normal = check_normality(combined_df['TruckCount'])

print(f"Car count normality: {car_normal}")
print(f"Bike count normality: {bike_normal}")
print(f"Bus count normality: {bus_normal}")
print(f"Truck count normality: {truck_normal}")


# In[115]:


# Check distribution after normalization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(combined_df['CarCount'], ax=axes[0, 0], kde=True, color='#1f77b4')
sns.histplot(combined_df['BikeCount'], ax=axes[0, 1], kde=True, color='#ff7f0e')
sns.histplot(combined_df['BusCount'], ax=axes[1, 0], kde=True, color='#2ca02c')
sns.histplot(combined_df['TruckCount'], ax=axes[1, 1], kde=True, color='#d62728')
axes[0, 0].set_title('Normalized Car Count')
axes[0, 1].set_title('Normalized Bike Count')
axes[1, 0].set_title('Normalized Bus Count')
axes[1, 1].set_title('Normalized Truck Count')
plt.tight_layout()
plt.show()


# In[116]:


# Prepare the features and target
X = combined_df.drop(columns=['Traffic Situation'])
y = combined_df['Traffic Situation']


# In[117]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[118]:


# Preprocessing pipeline for numeric and categorical features
numeric_features = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour']
categorical_features = ['Time', 'Date', 'Day of the week', 'Source', 'Weekend']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


# In[119]:


# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)


# In[120]:


# Make predictions
y_pred = model.predict(X_test)


# In[121]:


# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[122]:


# Inverse transform the normalized values to their original scale
inverse_transformed_data = scaler.inverse_transform(combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])
combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = inverse_transformed_data


# In[123]:


# Check some predictions with the inverse-transformed data
input_features = X_test.iloc[0].to_dict()
input_features_df = pd.DataFrame([input_features])

predicted_output = model.predict(input_features_df)
print(f"Input features: {input_features}")
print(f"Predicted output for input features: {predicted_output}")
