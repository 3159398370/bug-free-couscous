# Import necessary libraries
import sys

import plotly.graph_objects as go

from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, LabelEncoder
# 导入必要的库

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
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score




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
    ]
)

# 定义模型及其管道
models = {
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=42))]),

    'XGBoost': Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', xgb.XGBClassifier(random_state=42))]),

    'Support Vector Machine': Pipeline(steps=[('preprocessor', preprocessor),
                                              ('classifier', SVC(random_state=42, probability=True))]),

    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', GradientBoostingClassifier(random_state=42))]),

    'Extra Trees': Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', ExtraTreesClassifier(random_state=42))]),

    'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', DecisionTreeClassifier(random_state=42))]),

    'CatBoost': CatBoostClassifier(iterations=1000,
                                   learning_rate=0.1,
                                   depth=6,
                                   random_seed=42,
                                   cat_features=categorical_features,
                                   verbose=0)
}

# 训练和评估每个模型
results = []

for model_name, model in models.items():
    # 对于 CatBoost，需要单独训练
    if model_name == 'CatBoost':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 评估模型表现
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# 将结果转化为 DataFrame
results_df = pd.DataFrame(results)


    # 创建条形图可视化模型的性能
fig = go.Figure()

# 添加各指标的条形图
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    fig.add_trace(go.Bar(
        x=results_df['Model'],
        y=results_df[metric],
        name=metric,
        text=results_df[metric].apply(lambda x: f"{x:.4f}"),  # 转换为四位小数格式
        textposition='auto'  # 显示值在条形上
    ))

# 更新图形布局
fig.update_layout(
    title='Model Evaluation Metrics',  # 图标题
    xaxis_title='Models',              # X 轴标题
    yaxis_title='Scores',              # Y 轴标题
    barmode='group',                   # 条形图并排显示
    yaxis=dict(range=[0, 1]),          # Y 轴范围设置，从 0 到 1
    template='plotly_white'            # 白色背景样式
)

# 显示图形
fig.show()



# 分类报告字符串
report_texts = []
# 输出详细分类报告（可选）
for model_name, model in models.items():
    if model_name != 'CatBoost':
        y_pred = model.predict(X_test)
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, y_pred))
        report_texts.append(f"\nClassification Report for {model_name}:")
        report_texts.append(classification_report(y_test, y_pred))

# 将报告写入文件
with open(model_analysis_report, 'w') as f:
    for report in report_texts:
        f.write(report + "\n" + "="*50 + "\n")  # 分隔线可以帮助更清楚地区分不同报告