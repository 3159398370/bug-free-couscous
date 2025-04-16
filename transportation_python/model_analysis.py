# Import necessary libraries
import sys

import plotly.graph_objects as go

from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, LabelEncoder
# 导入必要的库
import numpy as np
import pandas as pd




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

import xgboost as xgb

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


model_analysis_report = sys.argv[2] # 新增model_type变量

input_train_filepath = sys.argv[1]
print(input_train_filepath)

print(model_analysis_report)

combined_df = pd.read_csv(input_train_filepath)
# 使用示例：
combined_df = preprocess_traffic_data(combined_df)


X = combined_df.drop(columns=['Traffic Situation'])
y = combined_df['Traffic Situation']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Preprocessing pipeline for numeric and categorical features
numeric_features = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Hour']
categorical_features = ['Time', 'Date', 'Day of the week', 'Weekend']





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

et_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ExtraTreesClassifier(random_state=42))
])

# 添加CatBoost模型（不用预处理步骤，因为CatBoost能够处理类别特征）
catBoost_model = CatBoostClassifier(iterations=1000,
                                    learning_rate=0.1,
                                    depth=6,
                                    random_seed=42,
                                    cat_features=categorical_features,
                                    verbose=0)  # verbose=0 以减少输出
# 训练梯度提升模型
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 训练随机森林模型
rf_model.fit(X_train, y_train)

# 训练XGBoost模型
xgb_model.fit(X_train, y_train)

# 训练支持向量机模型
svm_model.fit(X_train, y_train)

# 训练梯度提升模型
gb_model.fit(X_train, y_train)

et_model.fit(X_train, y_train)

catBoost_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# 对测试集进行预测
rf_y_pred = rf_model.predict(X_test)
xgb_y_pred = xgb_model.predict(X_test)
svm_y_pred = svm_model.predict(X_test)
gb_y_pred = gb_model.predict(X_test)
dt_y_pred = dt_model.predict(X_test)
catBoost_y_pred = catBoost_model.predict(X_test)
et_y_pred=et_model.predict(X_test)

#
# # 评估模型表现
# print("Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))
# print("Random Forest Classification Report:")
# print(classification_report(y_test, rf_y_pred))
#
# print("XGBoost Model Accuracy:", accuracy_score(y_test, xgb_y_pred))
# print("XGBoost Classification Report:")
# print(classification_report(y_test, xgb_y_pred))
#
# print("Support Vector Machine Model Accuracy:", accuracy_score(y_test, svm_y_pred))
# print("Support Vector Machine Classification Report:")
# print(classification_report(y_test, svm_y_pred))
#
# print("Gradient Boosting Model Accuracy:", accuracy_score(y_test, gb_y_pred))
# print("Gradient Boosting Classification Report:")
# print(classification_report(y_test, gb_y_pred))
#
#
# print("DT Model Accuracy:", accuracy_score(y_test, gb_y_pred))
# print("DT Classification Report:")
# print(classification_report(y_test, dt_y_pred))
#
#
# print("ET Model Accuracy:", accuracy_score(y_test, gb_y_pred))
# print("ET Classification Report:")
# print(classification_report(y_test, et_y_pred))
#
# print("catboost Model Accuracy:", accuracy_score(y_test, gb_y_pred))
# print("catboost Classification Report:")
# print(classification_report(y_test, catBoost_y_pred))



# In[125]:


from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score

results = {
    'Model': ['Random Forest', 'XGBoost', 'Support Vector Machine', 'Gradient Boosting','DT','ET','catboost'],
    'Accuracy': [
        accuracy_score(y_test, rf_y_pred),
        accuracy_score(y_test, xgb_y_pred),
        accuracy_score(y_test, svm_y_pred),
        accuracy_score(y_test, gb_y_pred),
        accuracy_score(y_test, dt_y_pred),
        accuracy_score(y_test, et_y_pred),
        accuracy_score(y_test, catBoost_y_pred)

    ],
    'Precision': [
        precision_score(y_test, rf_y_pred, average='weighted'),
        precision_score(y_test, xgb_y_pred, average='weighted'),
        precision_score(y_test, svm_y_pred, average='weighted'),
        precision_score(y_test, gb_y_pred, average='weighted'),
        precision_score(y_test, dt_y_pred, average='weighted'),
        precision_score(y_test, et_y_pred, average='weighted'),
        precision_score(y_test, catBoost_y_pred, average='weighted')
    ],
    'Recall': [
        recall_score(y_test, rf_y_pred, average='weighted'),
        recall_score(y_test, xgb_y_pred, average='weighted'),
        recall_score(y_test, svm_y_pred, average='weighted'),
        recall_score(y_test, gb_y_pred, average='weighted'),
        recall_score(y_test, dt_y_pred, average='weighted'),
        recall_score(y_test, et_y_pred, average='weighted'),
        recall_score(y_test, catBoost_y_pred, average='weighted')
    ],
    'F1 Score': [
        f1_score(y_test, rf_y_pred, average='weighted'),
        f1_score(y_test, xgb_y_pred, average='weighted'),
        f1_score(y_test, svm_y_pred, average='weighted'),
        f1_score(y_test, gb_y_pred, average='weighted'),
        f1_score(y_test, dt_y_pred, average='weighted'),
        f1_score(y_test, et_y_pred, average='weighted'),
        f1_score(y_test, catBoost_y_pred, average='weighted')
    ],
}

# 将结果转化为DataFrame
results_df = pd.DataFrame(results)
print(results_df)
# 输出详细分类报告（可选）
print("\n***************************************************************************:")


# 创建图形对象
fig = go.Figure()

# 添加各指标的条形图，并格式化值显示为小数点后四位
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    fig.add_trace(go.Bar(
        x=results_df['Model'],
        y=results_df[metric],
        name=metric,
        marker_color=np.random.choice(['#1f77b4', '#7fc1d7', '#a6cee3', '#ff7f0e', '#ffbb78'], len(results_df)),
        text=results_df[metric].apply(lambda x: f"{x:.4f}"),  # 转换为四位小数格式
        textposition='auto'  # 显示值在条形上
    ))

# 更新布局
fig.update_layout(
    title='Model Evaluation Metrics',
    xaxis_title='Models',
    yaxis_title='Scores',
    barmode='group',  # 在同一位置并排显示条形图
    yaxis=dict(range=[0, 1]),  # Y轴范围设置
    template='plotly_white'  # 白色背景
)

# 显示图形
fig.show()
print("\n***************************************************************************:")


# 分类报告字符串
report_texts = []

# 生成并保存分类报告
report_texts.append("Random Forest Classification Report:\n" + classification_report(y_test, rf_y_pred))
report_texts.append("XGBoost Classification Report:\n" + classification_report(y_test, xgb_y_pred))
report_texts.append("Support Vector Machine Classification Report:\n" + classification_report(y_test, svm_y_pred))
report_texts.append("Gradient Boosting Classification Report:\n" + classification_report(y_test, gb_y_pred))
report_texts.append("DT Classification Report:\n" + classification_report(y_test, dt_y_pred))
report_texts.append("ET Classification Report:\n" + classification_report(y_test, et_y_pred))
report_texts.append("catboost Classification Report:\n" + classification_report(y_test, catBoost_y_pred))

print("\n***************************************************************************:")
print(report_texts)

# 将报告写入文件
with open(model_analysis_report, 'w') as f:
    for report in report_texts:
        f.write(report + "\n" + "="*50 + "\n")  # 分隔线可以帮助更清楚地区分不同报告


# results_df.to_csv(model_analysis_report, index=False)


# In[126]:


# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], rf_y_prob[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # 绘制 ROC 曲线
# plt.figure(figsize=(10, 8))

# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], lw=2, label='Class {0} (AUC = {1:0.2f})'.format(classes[i], roc_auc[i]))

# # 绘制对角线
# plt.plot([0, 1], [0, 1], 'k--', lw=2)

# # 设置图形属性
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.grid()

# # 显示图形
# plt.show()

