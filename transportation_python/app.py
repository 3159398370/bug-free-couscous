from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS  # 导入 CORS 类
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

app = Flask(__name__)
CORS(app)  # 对应用启用 CORS，允许所有来源的跨域请求

# 示例数据，您需要替换成适当的 DataFrame
traffic_df = pd.read_csv(".\Traffic_analysis\Traffic.csv")
traffic_two_month_df = pd.read_csv(".\Traffic_analysis\TrafficTwoMonth.csv")
traffic_df['Source'] = 'OneMonth'
traffic_two_month_df['Source'] = 'TwoMonth'
combined_df = pd.concat([traffic_df, traffic_two_month_df], ignore_index=True)


@app.route('/')
def index():
    return "Welcome to the traffic prediction system API!"


@app.route('/favicon.ico')
def favicon():
    return '', 204  # 返回无内容响应


@app.route('/api/avg_vehicle_counts', methods=['GET'])
# def get_avg_vehicle_counts():
#     numeric_cols = combined_df.select_dtypes(include=['number']).columns
#     avg_time_counts = combined_df.groupby('Time')[numeric_cols].mean().reset_index()
#     result = avg_time_counts.to_dict(orient='records')
#
#     return jsonify(result)

def get_avg_vehicle_counts():
    """
    计算数据框中按 'Time' 分组后数值列的平均值，按时间排序后将结果以 JSON 格式返回。
    :param df: 输入的数据框
    :return: JSON 格式的结果
    """
    # 选择数值列
    numeric_cols = combined_df.select_dtypes(include=['number']).columns
    # 按 'Time' 分组并计算平均值
    avg_time_counts = combined_df.groupby('Time')[numeric_cols].mean().reset_index()
    # 确保 Time 列按时间顺序排序（字符串类型也能正确排序）
    avg_time_counts = avg_time_counts.sort_values(by='Time')
    # 将结果转换为字典列表
    result = avg_time_counts.to_dict(orient='records')
    # 返回 JSON 响应
    return jsonify(result)






@app.route('/api/vehicle_counts_by_week', methods=['GET'])
def vehicle_counts_by_week():
    # 按星期对数据进行分组并求和
    weekly_data = combined_df.groupby('Day of the week').agg({
        'CarCount': 'sum',
        'BikeCount': 'sum',
        'BusCount': 'sum',
        'TruckCount': 'sum'
    }).reset_index()

    # 将数据转换为字典列表
    result = []
    for _, row in weekly_data.iterrows():
        result.append({
            "day": row["Day of the week"],
            "CarCount": row["CarCount"],
            "BikeCount": row["BikeCount"],
            "BusCount": row["BusCount"],
            "TruckCount": row["TruckCount"]
        })

    return jsonify(result)



@app.route('/api/VehicleChart4', methods=['GET'])
def get_vehicle_counts():
    # 从数据中提取和聚合信息
    vehicle_counts = {
        'BikeCount': combined_df.groupby('Traffic Situation')['BikeCount'].sum().tolist(),
        'BusCount': combined_df.groupby('Traffic Situation')['BusCount'].sum().tolist(),
        'CarCount': combined_df.groupby('Traffic Situation')['CarCount'].sum().tolist(),
        'TruckCount': combined_df.groupby('Traffic Situation')['TruckCount'].sum().tolist()
    }

    # 翻译"Traffic Situation"
    traffic_situations = combined_df['Traffic Situation'].unique().tolist()
    translated_situations = [
        "高度拥堵" if situation == "low" else
        "正常" if situation == "normal" else
        "低车流量" if situation == "heavy" else
        "重度拥堵" if situation == "high" else situation
        for situation in traffic_situations
    ]

    vehicle_counts['Traffic Situation'] = translated_situations

    return jsonify(vehicle_counts)






if __name__ == '__main__':
    app.run(debug=True)