<template>
    <div>
        <div ref="chart" style="width: 1000px; height: 600px"></div>
    </div>
</template>

<script>
    import * as echarts from 'echarts'

    export default {
        name: 'VehicleChart',
        data() {
            return {
                chartInstance: null,
            };
        },
        methods: {
            fetchData() {
                fetch('http://127.0.0.1:5000/api/VehicleChart4') // 确保这个端点返回的格式正确
                    .then(response => response.json())
                    .then(data => {
                        console.log('获取的数据:', data); // 插入调试信息
                        this.drawChart(data);
                    })
                    .catch(error => {

                        console.error('获取数据时出错:', error);
                    });
            },
            drawChart(data) {
                const option = {
                    title: {
                        text: '',
                        left: 'center',
                    },
                    tooltip: {
                        trigger: 'axis',
                    },
                    legend: {
                        data: ['car Counts', 'bike Counts', 'bus Counts', 'truck Counts'],
                        top: '1%',
                    },
                    xAxis: {
                        type: 'category',
                        data: data['Traffic Situation'],
                        axisLabel: {
                            interval: 0,
                        },
                    },
                    yAxis: {
                        type: 'value',
                        name: '数量',
                    },
                    series: [
                        {
                            name: 'car Counts',
                            type: 'bar',
                            data: data['CarCount'],
                            itemStyle: {
                                color: 'rgba(31, 119, 180, 0.6)',
                            },
                        },
                        {
                            name: 'bike Counts',
                            type: 'bar',
                            data: data['BikeCount'],
                            itemStyle: {
                                color: 'rgba(255, 127, 14, 0.6)',
                            },
                        },
                        {
                            name: 'bus Counts',
                            type: 'bar',
                            data: data['BusCount'],
                            itemStyle: {
                                color: 'rgba(44, 160, 44, 0.6)',
                            },
                        },
                        {
                            name: 'truck Counts',
                            type: 'bar',
                            data: data['TruckCount'],
                            itemStyle: {
                                color: 'rgba(214, 39, 40, 0.6)',
                            },
                        },
                    ],
                };

                // 初始化或更新图表
                if (!this.chartInstance) {
                    this.chartInstance = echarts.init(this.$refs.chart);
                }
                this.chartInstance.setOption(option);
            },
        },
        mounted() {
            this.fetchData();
        },
    };
</script>