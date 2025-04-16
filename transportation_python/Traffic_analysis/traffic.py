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


# 加载训练数据集文件
# 参数说明：
# sys.argv[1] - 命令行传入的训练数据集文件路径，应为CSV格式
input_train_filepath = sys.argv[1]

# 读取完整流量数据到主DataFrame
# traffic_df将包含原始训练集的所有记录
traffic_df = pd.read_csv(input_train_filepath)

# 读取相同数据到新DataFrame用于后续两个月周期分析
# traffic_two_month_df将用于专门处理两个月时间跨度的分析任务
traffic_two_month_df = pd.read_csv(input_train_filepath)


# In[89]:


# 查看一个月交通数据集前10行样本
traffic_df.head(10)


# In[90]:


# 查看两个月交通数据集前10行样本
traffic_two_month_df.head(10)


# In[91]:


# 统计交通状况字段的类别分布
traffic_df['Traffic Situation'].value_counts()


# In[92]:


# 合并数据集操作说明：
# 1. 为原始数据集添加来源标识列
# 2. 垂直拼接两个数据集，重置索引
traffic_df['Source'] = 'OneMonth'
traffic_two_month_df['Source'] = 'TwoMonth'
combined_df = pd.concat([traffic_df, traffic_two_month_df], ignore_index=True)


# In[93]:


# 创建2x2子图布局，展示四类车辆数量分布直方图
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Car Counts", "Bike Counts", "Bus Counts", "Truck Counts")
)

# 添加四个子图的直方图数据
fig.add_trace(go.Histogram(x=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Histogram(x=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Histogram(x=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Histogram(x=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

# 设置整体图表样式
fig.update_layout(
    title_text='Distribution of Vehicle Counts',
    title_x=0.5,
    showlegend=False,
    template='plotly_white'
)
# 统一设置坐标轴标签
fig.update_xaxes(title_text="Count")
fig.update_yaxes(title_text="Frequency")
#fig.show()



# In[94]:


# 3. Distribution of traffic situations
# --------------------------------------------------
# 创建交通状况分布饼图
# 参数说明：
# - names: 使用'Traffic Situation'列作为分类维度
# - title: 图表主标题
# - color_discrete_sequence: 使用红蓝渐变色系
fig = px.pie(combined_df, names='Traffic Situation', title='Traffic Situation Distribution', color_discrete_sequence=px.colors.sequential.RdBu)
# 更新布局配置：
# - title_text: 标题文本
# - title_x: 标题水平居中
# - template: 使用白色背景模板
fig.update_layout(title_text='Traffic Situation Distribution', title_x=0.5, template='plotly_white')


# 4. Vehicle count vary by day of the week
# --------------------------------------------------
# 创建2x2子图矩阵展示不同车型周分布情况
# 子图标题：
# - 第一行：汽车/自行车数量按周分布
# - 第二行：公交车/卡车数量按周分布
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Day", "Bike Counts by Day", "Bus Counts by Day", "Truck Counts by Day"))

# 添加各子图数据：
# - 使用箱线图展示分布情况
# - 不同车型使用不同标识颜色
# - 通过row/col参数指定子图位置
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
fig.add_trace(go.Box(x=combined_df['Day of the week'], y=combined_df['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

# 全局布局配置：
# - 主标题居中显示
# - 隐藏图例
# - 设置坐标轴标签
fig.update_layout(title_text='Vehicle Counts by Day of the Week', title_x=0.5, showlegend=False, template='plotly_white')
fig.update_xaxes(title_text="Day of the Week")
fig.update_yaxes(title_text="Count")


# 5-8. Relationship between vehicle counts and traffic situation
# --------------------------------------------------
# 分析不同交通状况下的车型数量分布
# 子图结构：
# - 2x2矩阵展示四种车型
# - 使用箱线图对比不同交通状况下的数量分布
fig = make_subplots(rows=2, cols=2, subplot_titles=("Car Counts by Traffic Situation", "Bike Counts by Traffic Situation", "Bus Counts by Traffic Situation", "Truck Counts by Traffic Situation"))

# 添加子图数据（参数配置与图4类似）


# 9. Total vehicle count by traffic situation
# --------------------------------------------------
# 展示不同交通状况下的总车流量分布
# 图表类型：箱线图
# 参数说明：
# - x: 交通状况分类
# - y: 总车流量数值
fig = px.box(combined_df, x='Traffic Situation', y='Total', title='Total Vehicle Count by Traffic Situation', color_discrete_sequence=px.colors.sequential.RdBu)


# 13. Correlations between different vehicle types
# --------------------------------------------------
# 计算车型数量间的相关系数矩阵
# 分析维度：
# - 包含汽车、自行车、公交车、卡车及总车流量
# 可视化方式：
# - 使用冷热色系突出相关性强度
# - 显示具体相关系数值
corr_matrix = combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# 16. Distribution of vehicle counts between weekdays and weekends
# --------------------------------------------------
# 对比工作日与周末的车型数量差异
# 数据准备：
# - 新增'Weekend'布尔列标识周末
# 可视化方式：
# - 2x2矩阵箱线图展示各车型分布
combined_df['Weekend'] = combined_df['Day of the week'].isin(['Saturday', 'Sunday'])


# 18. Distribution of traffic situations by hour
# --------------------------------------------------
# 分析不同时段的交通状况分布
# 图表类型：箱线图
# 参数说明：
# - x: 小时数值型数据
# - y: 交通状况分类数据


# 20. Distribution of total vehicle counts for each day of the week
# --------------------------------------------------
# 展示周各天的总车流量分布
# 图表类型：箱线图
# 样式配置：
# - 使用单一紫色系配色
# - 优化坐标轴标签显示
fig = px.box(combined_df, x='Day of the week', y='Total', title='Total Vehicle Count by Day of the Week', color_discrete_sequence=['#9467bd'])

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
    """
    使用IQR方法从指定列中移除异常值

    参数：
        df (pd.DataFrame): 包含数据的输入DataFrame
        columns (list): 需要处理异常值的列名列表

    返回值：
        pd.DataFrame: 移除异常值后的新DataFrame

    处理逻辑：
        1. 对每个指定列计算IQR（四分位距）
        2. 过滤超出[Q1-1.5IQR, Q3+1.5IQR]范围的值
        3. 迭代处理每个指定列，逐步过滤数据
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# 需要处理异常值的车辆计数特征列
vehicle_counts = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
# 对合并后的数据集进行异常值移除
combined_df = remove_outliers(combined_df, vehicle_counts)


# 检查数据质量
# Check for missing values and duplicates
print("Missing values in each column:")
print(combined_df.isnull().sum())

print(f"Number of duplicate rows: {combined_df.duplicated().sum()}")


# 可视化异常值处理效果
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
    '''使用Shapiro-Wilk检验验证数据正态性

    Args:
        data: 待检验的数值型数组/列表，需为一维数据序列

    Returns:
        bool: 当p值>0.05时返回True，表示不能拒绝原假设（数据服从正态分布）
    '''
    stat, p = shapiro(data)
    return p > 0.05


# 原始数据正态性检验流程
print("Normality check before normalization:")
# 对四种车辆类型计数分别进行正态性检验
car_normal = check_normality(combined_df['CarCount'])
bike_normal = check_normality(combined_df['BikeCount'])
bus_normal = check_normality(combined_df['BusCount'])
truck_normal = check_normality(combined_df['TruckCount'])

print(f"Car count normality: {car_normal}")
print(f"Bike count normality: {bike_normal}")
print(f"Bus count normality: {bus_normal}")
print(f"Truck count normality: {truck_normal}")

# 使用分位数变换进行数据标准化
# 设置输出分布为正态分布，通过分位数映射强制数据服从高斯分布
scaler = QuantileTransformer(output_distribution='normal')
combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = scaler.fit_transform(combined_df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])

# 标准化后数据正态性验证
print("Normality check after normalization:")
# 对标准化后的四种车辆类型计数重新进行正态性检验
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


# In[124]:


# 导入必要的库  
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report  
import xgboost as xgb  

# 假设你的数据是一个DataFrame，命名为data  
# data = pd.read_csv('your_data.csv')  

# 定义特征和目标变量  
# X = data.drop(columns='target_column_name')  
# y = data['target_column_name']  

# 将数据集分成训练集和测试集  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# 定义数值特征和类别特征（根据你的数据集进行调整）  
# numeric_features = ['numerical_column1', 'numerical_column2', ...]  
# categorical_features = ['categorical_column1', 'categorical_column2', ...]  

# 创建预处理管道  
preprocessor = ColumnTransformer(  
    transformers=[  
        ('num', StandardScaler(), numeric_features),  
        ('cat', OneHotEncoder(), categorical_features)  
    ])  

# 定义各个模型管道  
rf_model = Pipeline(steps=[  
    ('preprocessor', preprocessor),  
    ('classifier', RandomForestClassifier(random_state=42))  
])  

xgb_model = Pipeline(steps=[  
    ('preprocessor', preprocessor),  
    ('classifier', xgb.XGBClassifier(random_state=42))  
])  

svm_model = Pipeline(steps=[  
    ('preprocessor', preprocessor),  
    ('classifier', SVC(random_state=42,probability=True))  
])  

gb_model = Pipeline(steps=[  
    ('preprocessor', preprocessor),  
    ('classifier', GradientBoostingClassifier(random_state=42))  
])  

# 训练随机森林模型  
rf_model.fit(X_train, y_train)  

# 训练XGBoost模型  
xgb_model.fit(X_train, y_train)  

# 训练支持向量机模型  
svm_model.fit(X_train, y_train)  

# 训练梯度提升模型  
gb_model.fit(X_train, y_train)  

# 对测试集进行预测  
rf_y_pred = rf_model.predict(X_test)  
xgb_y_pred = xgb_model.predict(X_test)  
svm_y_pred = svm_model.predict(X_test)  
gb_y_pred = gb_model.predict(X_test)  

# 评估模型表现  
print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))  
print("Random Forest Classification Report:")  
print(classification_report(y_test, rf_y_pred))  

print("XGBoost Model Accuracy:", accuracy_score(y_test, xgb_y_pred))  
print("XGBoost Classification Report:")  
print(classification_report(y_test, xgb_y_pred))  

print("Support Vector Machine Model Accuracy:", accuracy_score(y_test, svm_y_pred))  
print("Support Vector Machine Classification Report:")  
print(classification_report(y_test, svm_y_pred))  

print("Gradient Boosting Model Accuracy:", accuracy_score(y_test, gb_y_pred))  
print("Gradient Boosting Classification Report:")  
print(classification_report(y_test, gb_y_pred))


# In[125]:


from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score

results = {  
    'Model': ['Random Forest', 'XGBoost', 'Support Vector Machine', 'Gradient Boosting'],  
    'Accuracy': [  
        accuracy_score(y_test, rf_y_pred),  
        accuracy_score(y_test, xgb_y_pred),  
        accuracy_score(y_test, svm_y_pred),  
        accuracy_score(y_test, gb_y_pred)  
    ],  
    'Precision': [  
        precision_score(y_test, rf_y_pred, average='weighted'),  
        precision_score(y_test, xgb_y_pred, average='weighted'),  
        precision_score(y_test, svm_y_pred, average='weighted'),  
        precision_score(y_test, gb_y_pred, average='weighted')  
    ],  
    'Recall': [  
        recall_score(y_test, rf_y_pred, average='weighted'),  
        recall_score(y_test, xgb_y_pred, average='weighted'),  
        recall_score(y_test, svm_y_pred, average='weighted'),  
        recall_score(y_test, gb_y_pred, average='weighted')  
    ],  
    'F1 Score': [  
        f1_score(y_test, rf_y_pred, average='weighted'),  
        f1_score(y_test, xgb_y_pred, average='weighted'),  
        f1_score(y_test, svm_y_pred, average='weighted'),  
        f1_score(y_test, gb_y_pred, average='weighted')  
    ],  
}  

# 将结果转化为DataFrame  
results_df = pd.DataFrame(results)  

# 打印模型比较结果  
print(results_df)  

# 输出详细分类报告（可选）  
print("\nClassification Reports:")  
print("Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred))  
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_y_pred))  
print("Support Vector Machine Classification Report:\n", classification_report(y_test, svm_y_pred))  
print("Gradient Boosting Classification Report:\n", classification_report(y_test, gb_y_pred))  


# In[126]:

