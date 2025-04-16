
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



# In[125]:


# 导入必要的评估指标和库
from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score

# 定义模型评估结果数据字典
# 包含7个模型的四类评估指标计算结果：
# - Model：模型名称列表
# - Accuracy：测试集准确率列表
# - Precision：加权精确率列表
# - Recall：加权召回率列表
# - F1 Score：加权F1分数列表
results = {
    'Model': ['Random Forest', 'XGBoost', 'Support Vector Machine', 'Gradient Boosting','DT','ET','catboost'],
    'Accuracy': [
        accuracy_score(y_test, rf_y_pred),

    ],

}

# 将结果字典转换为DataFrame便于分析展示
# 转换后的二维表格结构方便后续打印和可视化处理
results_df = pd.DataFrame(results)
print(results_df)

# 创建交互式可视化图表对象
# 使用Plotly库绘制多指标对比条形图
fig = go.Figure()

# 为每个评估指标添加条形图轨迹
# 特征说明：
# - x轴为模型名称，y轴为指标分数
# - 自动显示格式化后的4位小数数值
# - 随机选择预定义颜色方案
# - 采用分组显示模式便于对比
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    fig.add_trace(go.Bar(
        x=results_df['Model'],
        y=results_df[metric],
        # 参数配置...
    ))

# 配置图表布局参数
# 包含标题设置、坐标轴标签、显示范围限制
# 使用白色主题模板，启用分组条形模式
fig.update_layout(
    title='Model Evaluation Metrics',
    # 其他布局参数...
)

# 生成详细分类报告集合
# 包含各模型的precision/recall/f1-score等细粒度指标
report_texts = []
report_texts.append("Random Forest Classification Report:\n" + classification_report(y_test, rf_y_pred))
# 其他模型报告生成逻辑类似...

# 将分类报告写入指定文件
# 文件格式：每个报告之间用等号线分隔
with open(model_analysis_report, 'w') as f:
    for report in report_texts:
        f.write(report + "\n" + "="*50 + "\n")
