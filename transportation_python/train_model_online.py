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
from sklearn.linear_model import LogisticRegression
 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score, classification_report  
import xgboost as xgb
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier



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



def train_traffic_data(combined_df,model_type):
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
            ('cat',OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    train_model = None



  

  

    if model_type == 'modelRF':
            # 定义各个模型管道  
        train_model = Pipeline(steps=[  
            ('preprocessor', preprocessor),  
            ('classifier', RandomForestClassifier(random_state=42))  
        ])
        # 训练随机森林模型  
        train_model.fit(X_train, y_train)
        # 评估模型表现  
        rf_y_pred = train_model.predict(X_test)
        print("****************************************************************")
        print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))  
        print("Random Forest Classification Report:")  
        print(classification_report(y_test, rf_y_pred))  
        print("****************************************************************")
        report = classification_report(y_test, rf_y_pred, output_dict=True)
        dump(train_model, model_save_path)

    if model_type == 'modelXGBoost':
        xgb_model = Pipeline(steps=[  
            ('preprocessor', preprocessor),  
            ('classifier', xgb.XGBClassifier(random_state=42))  
        ])      
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
        svm_model = Pipeline(steps=[  
            ('preprocessor', preprocessor),  
            ('classifier', SVC(random_state=42,probability=True))  
        ])
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
        gb_model = Pipeline(steps=[  
            ('preprocessor', preprocessor),  
            ('classifier', GradientBoostingClassifier(random_state=42))  
        ])
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

    if model_type == 'modelLR':
        # 训练梯度提升模型  
        lr_model = Pipeline(steps=[  
            ('preprocessor', preprocessor),  
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))  
        ])
        lr_model.fit(X_train, y_train) 
        gb_y_pred = lr_model.predict(X_test)
        print("****************************************************************")
        print("LR Accuracy:", accuracy_score(y_test, gb_y_pred))  
        print("LR Classification Report:")  
        classification_report(y_test, gb_y_pred)
        print(classification_report(y_test, gb_y_pred))
        print("****************************************************************")
        report = classification_report(y_test, gb_y_pred, output_dict=True)
        dump(lr_model,model_save_path)



    if model_type == 'modelDT':
        # 训练梯度提升模型  
        dt_model = Pipeline(steps=[  
            ('preprocessor', preprocessor),  
            ('classifier', DecisionTreeClassifier(random_state=42))  
        ]) 
        dt_model.fit(X_train, y_train) 
        dt_y_pred = dt_model.predict(X_test)
        print("****************************************************************")
        print("LR Accuracy:", accuracy_score(y_test, dt_y_pred))  
        print("LR Classification Report:")  
        classification_report(y_test, dt_y_pred)
        print(classification_report(y_test, dt_y_pred))
        print("****************************************************************")
        report = classification_report(y_test, dt_y_pred, output_dict=True)
        dump(dt_model,model_save_path)
       # 添加决策树模型

    if model_type == 'catBoost':
        # 添加CatBoost模型（不用预处理步骤，因为CatBoost能够处理类别特征）
        catBoost_model = CatBoostClassifier(iterations=1000,
                                       learning_rate=0.1,
                                       depth=6,
                                       random_seed=42,
                                       cat_features=categorical_features,
                                       verbose=0)  # verbose=0 以减少输出
        catBoost_model.fit(X_train, y_train)
        cat_y_pred = catBoost_model.predict(X_test)
        print("****************************************************************")
        print("LR Accuracy:", accuracy_score(y_test, cat_y_pred))
        print("LR Classification Report:")
        classification_report(y_test, cat_y_pred)
        print(classification_report(y_test, cat_y_pred))
        print("****************************************************************")
        report = classification_report(y_test, cat_y_pred, output_dict=True)
        dump(catBoost_model,model_save_path)


    if model_type == 'modelET':
            # 添加Extra Trees模型
        et_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', ExtraTreesClassifier(random_state=42))
        ])

        et_model.fit(X_train, y_train)
        et_y_pred = et_model.predict(X_test)
        print("****************************************************************")
        print("LR Accuracy:", accuracy_score(y_test, et_y_pred))
        print("LR Classification Report:")
        classification_report(y_test, et_y_pred)
        print(classification_report(y_test, et_y_pred))
        print("****************************************************************")
        report = classification_report(y_test, et_y_pred, output_dict=True)
        dump(et_model,model_save_path)

    #
    # # 提取 precision、recall 和 f1-score
    # labels = list(report.keys())[:-3]  # 排除 'accuracy', 'macro avg', 'weighted avg'
    # precision = [report[label]['precision'] for label in labels]
    # recall = [report[label]['recall'] for label in labels]
    # f1_score = [report[label]['f1-score'] for label in labels]
    #
    # # 绘制折线图
    # x = np.arange(len(labels))
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, precision, marker='o', label='Precision', color='blue')
    # plt.plot(x, recall, marker='o', label='Recall', color='green')
    # plt.plot(x, f1_score, marker='o', label='F1 Score', color='orange')
    #
    # # 添加标签和标题
    # plt.xticks(x, labels)
    # plt.xlabel('Classes')
    # plt.ylabel('Scores')
    # plt.title('Classification Metrics')
    # plt.ylim(0, 1)
    # plt.grid()
    # plt.legend()
    #
    # # 展示图像
    # plt.show()
    # 提取 precision、recall 和 f1-score
    labels = list(report.keys())[:-3]  # 排除 'accuracy', 'macro avg', 'weighted avg'
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1_score = [report[label]['f1-score'] for label in labels]

    # 绘制交互式条形图
    fig = go.Figure()

    # 添加条形图数据
    fig.add_trace(go.Bar(
        x=labels,
        y=precision,
        name='Precision',
        marker_color='#1f77b4'  # 深蓝色
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=recall,
        name='Recall',
        marker_color='#7fc1d7'  # 浅蓝色
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=f1_score,
        name='F1 Score',
        marker_color='#a6cee3'  # 更浅的蓝色
    ))

    # 更新布局
    fig.update_layout(
        title='Classification Metrics',
        xaxis_title='Classes',
        yaxis_title='Scores',
        barmode='group',  # 将条形并排显示
        yaxis=dict(range=[0, 1]),  # 设置 y 轴范围
    )

    # 显示图形
    fig.show()

# 显示图形
    plt.show()
#
# return model_save_path

train_traffic_data(combined_df,model_type)