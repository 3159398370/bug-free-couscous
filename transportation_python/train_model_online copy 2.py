# Import necessary libraries
import sys
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

 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report  
import xgboost as xgb
from joblib import dump


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


def preprocess_traffic_data(combined_df):  
    """  
    对交通数据进行预处理。  
    
    :param combined_df: 包含交通数据的 DataFrame  
    :param vehicle_counts: 需要标准化的车辆计数列名列表  
    :return: 经过预处理的 DataFrame  
    """  
    vehicle_counts = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
    # 1. 将 'Traffic Situation' 列转换为分类变量并提取编码  
    combined_df['Traffic Situation'] = combined_df['Traffic Situation'].astype('category').cat.codes  
    
    # 2. 从 'Time' 列提取小时信息  
    combined_df['Hour'] = pd.to_datetime(combined_df['Time'], format='%I:%M:%S %p').dt.hour  
    
    # 3. 创建一个布尔列 'Weekend'，判断是否为周末  
    combined_df['Weekend'] = combined_df['Day of the week'].isin(['Saturday', 'Sunday'])  
    
    # 4. 移除异常值  
    combined_df = remove_outliers(combined_df, vehicle_counts)  
    
    # 5. 使用 QuantileTransformer 进行标准化  
    scaler = QuantileTransformer(output_distribution='normal')  
    combined_df[vehicle_counts] = scaler.fit_transform(combined_df[vehicle_counts])  
    
    return combined_df  


model_type = sys.argv[3] # 新增model_type变量
model_save_path = sys.argv[2]
input_train_filepath = sys.argv[1]

combined_df = pd.read_csv(input_train_filepath)
# 使用示例：  
combined_df = preprocess_traffic_data(combined_df) 



def preprocess_traffic_data(combined_df,model_type):  
    # Prepare the features and target
    X = combined_df.drop(columns=['Traffic Situation'])
    y = combined_df['Traffic Situation']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Preprocessing pipeline for numeric and categorical features
    numeric_features = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour']
    categorical_features = ['Time', 'Date', 'Day of the week', 'Weekend']
    report = None
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

    if model_type == 'modelRF':
        # 训练随机森林模型  
        rf_model.fit(X_train, y_train)
        # 评估模型表现  
        rf_y_pred = rf_model.predict(X_test)
        print("****************************************************************")
        print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))  
        print("Random Forest Classification Report:")  
        print(classification_report(y_test, rf_y_pred))  
        print("****************************************************************")
        report = classification_report(y_test, rf_y_pred, output_dict=True)
        dump(rf_model, model_save_path)

    if model_type == 'modelXGBoost':
        # 训练XGBoost模型  
        xgb_model.fit(X_train, y_train) 
        xgb_y_pred = xgb_model.predict(X_test)
        print("****************************************************************")
        print("XGBoost Model Accuracy:", accuracy_score(y_test, xgb_y_pred))  
        print("XGBoost Classification Report:")  
        print(classification_report(y_test, xgb_y_pred))
        print("****************************************************************")
        report = classification_report(y_test, xgb_y_pred, output_dict=True)
        dump(xgb_model, model_save_path)

    if model_type == 'modelSVM':
        # 训练支持向量机模型  
        svm_model.fit(X_train, y_train) 
        svm_y_pred = svm_model.predict(X_test)
        print("****************************************************************")
        print("Support Vector Machine Model Accuracy:", accuracy_score(y_test, svm_y_pred))  
        print("Support Vector Machine Classification Report:")  
        print(classification_report(y_test, svm_y_pred)) 
        print("****************************************************************")
        report = classification_report(y_test, svm_y_pred, output_dict=True)
        dump(svm_model,model_save_path)

    if model_type == 'modelGB':
        # 训练梯度提升模型  
        gb_model.fit(X_train, y_train) 
        gb_y_pred = gb_model.predict(X_test)
        print("****************************************************************")
        print("Gradient Boosting Model Accuracy:", accuracy_score(y_test, gb_y_pred))  
        print("Gradient Boosting Classification Report:")  
        classification_report(y_test, gb_y_pred)
        print(classification_report(y_test, gb_y_pred))
        print("****************************************************************")
        report = classification_report(y_test, gb_y_pred, output_dict=True)
        dump(gb_model,model_save_path)




    # 提取 precision、recall 和 f1-score  
    labels = list(report.keys())[:-3]  # 排除 'accuracy', 'macro avg', 'weighted avg'  
    precision = [report[label]['precision'] for label in labels]  
    recall = [report[label]['recall'] for label in labels]  
    f1_score = [report[label]['f1-score'] for label in labels]  

    # 绘制折线图  
    x = np.arange(len(labels))  

    plt.figure(figsize=(10, 6))  
    plt.plot(x, precision, marker='o', label='Precision', color='blue')  
    plt.plot(x, recall, marker='o', label='Recall', color='green')  
    plt.plot(x, f1_score, marker='o', label='F1 Score', color='orange')  

    # 添加标签和标题  
    plt.xticks(x, labels)  
    plt.xlabel('Classes')  
    plt.ylabel('Scores')  
    plt.title('Classification Metrics')  
    plt.ylim(0.5, 1)
    plt.grid()  
    plt.legend()  
    
    # 展示图像  
    plt.show()  

    return model_save_path

preprocess_traffic_data(combined_df,model_type)