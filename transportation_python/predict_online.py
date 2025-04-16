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
import joblib
import pandas as pd
import sys
import json
import os
# 获取前端发送过来的文件地址
predict_file_path = sys.argv[1]
output_predict_file_name = sys.argv[2]



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

combined_df = pd.read_csv(predict_file_path)
# 使用示例：
combined_df = preprocess_traffic_data(combined_df)


loaded_model = joblib.load('H\\mach\\transportation_predict_system\\transportation_python\\Traffic_analysis\\DTmodel.jobli')



# 准备特征
X_new = combined_df.drop(columns=['Traffic Situation'])  # 假设新数据没有目标变量



# 进行预测
y_pred = loaded_model.predict(X_new)





# 打印预测结果
print("Predictions:", y_pred)

result = {}
for i, pred in enumerate(y_pred):
    result[str(i)] = int(pred)



# 将结果写入 JSON 文件
with open(output_predict_file_name, 'w') as f:
    json.dump(result, f)

print(f"Predictions saved to {output_predict_file_name}")




#
# X_test =  combined_df.drop(columns=['Traffic Situation'])
# 加载模型
# loaded_model = joblib.load('D:\\tecachworkspace\\wanganqi\\transportation_predict_system\\transportation_python\\Traffic_analysis\\RFmodel.jobli')
#
# # 使用加载的模型进行预测
# y_pred = loaded_model.predict(X_test)
#
# # 打印预测结果
# print("Predictions:", y_pred)
#
# # 保存预测结果到 JSON 文件
# predictions_dict = {
#     'predictions': y_pred.tolist()  # 将 numpy 数组转换为 Python 列表
# }
#
# # 指定保存路径
#
# with open(output_predict_file_name, 'w') as json_file:
#     json.dump(predictions_dict, json_file, indent=4)  # 将预测结果写入 JSON 文件
#
# print(f"Predictions saved to {output_predict_file_name}")





