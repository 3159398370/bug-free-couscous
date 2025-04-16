<template>
  <div style="color: #666;font-size: 14px;">
    <div style="padding-bottom: 20px">
      <h1 style="text-align: center"><b>亲爱的{{ user.nickname }}，欢迎登录AI多模型的交通情况预测系统</b></h1>
    </div>
    <el-row :gutter="10" style="margin-bottom: 5px">
      <el-col :span="6">
        <el-card style="color: #409EFF">
          <div><i class="el-icon-user-solid" />当前用户总数</div>
          <div style="padding: 10px 0; text-align: center; font-weight: bold" v-text="total"></div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card style="color: #409EFF;margin-bottom:50px">
          <div><i class="el-icon-user-solid" />拥堵总数</div>
          <div style="padding: 10px 0; text-align: center; font-weight: bold" v-text="total2"></div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card style="color: #409EFF;margin-bottom:50px">
          <div><i class="el-icon-user-solid" />今日拥堵预测总数</div>
          <div style="padding: 10px 0; text-align: center; font-weight: bold" v-text="total1"></div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card style="color: #409EFF;margin-bottom:50px">
          <div><i class="el-icon-user-solid" />当前重度拥堵情况</div>
          <div style="padding: 10px 0; text-align: center; font-weight: bold">{{ total3 }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20">
      <el-col :span="12">
        <el-card shadow="hover" style="width: 100%;">
          <div slot="header" class="clearfix">
            <span>交通拥堵分类图</span>
          </div>
          <el-col :span="12">
            <div id="tu" style="width: 600px; height: 600px"></div>
          </el-col>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover" style="width: 100%;">
          <div slot="header" class="clearfix">
            <span>随时间变化的平均车辆数量</span>
          </div>
          <el-col :span="12">
            <div id="tu1" style="width: 1000px; height: 600px"></div>
          </el-col>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover" style="width: 100%;">
          <div slot="header" class="clearfix">
            <span>车辆计数按星期展示</span>
          </div>
          <el-col :span="12">
            <div id="weeklyChart" style="width: 1000px; height: 600px"></div>
          </el-col>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="hover" style="width: 100%;">
          <div slot="header" class="clearfix">
            <span>车辆数量VS交通情况分类</span>
          </div>
          <el-col :span="12">
<!--             <div id="weeklyChart1" style="width: 600px; height: 600px"></div>-->
                <VehicleChart />
          </el-col>
        </el-card>
      </el-col>
    </el-row>

  </div>
</template>
<script>
  import VehicleChart from "@/components/VehicleChart";
  import * as echarts from 'echarts'
  import axios from 'axios';  // 导入 axios

  export default {
    name: "Home",
      components: {
        VehicleChart // 注册组件
      },
      data() {
      return {
        user: localStorage.getItem("user") ? JSON.parse(localStorage.getItem("user")) : {},
        total: 0,
        total1: 0,
        total2: 0,
        total3:0,

      }

    },
    created() {
      this.load(),
              this.load1(),
              this.load2(),
              this.load3()
    },
    methods: {
      load() {
        this.request.get("/user/totle", {
        }).then(res => {
          this.total = res.data
        })
      },
      load2() {
        this.request.get("/echarts/totle").then(res => {
          this.total2=res.data
        })
      },
      load1() {
        this.request.get("/echarts/totle1", {
        }).then(res => {
          this.total1 = res.data
        })
      },
      load3() {
        this.request.get("/echarts/totle3").then(res => {
          this.total3=res.data
        })
      },


      loadVehicleCountsByWeek() {
        // 请求后端的车辆计数按星期的API
        axios.get('http://127.0.0.1:5000/api/vehicle_counts_by_week')
                .then(response => {
                  const data = response.data;
                  this.renderWeeklyChart(data); // 调用渲染函数
                })
                .catch(error => {
                  console.error('Error fetching vehicle counts by week:', error);
                });
      },

      renderWeeklyChart(data) {
        // 提取星期和各类车辆计数数据
        const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]; // 完整的星期数组
        const carCounts = new Array(days.length).fill(0); // 初始化车辆计数数组
        const bikeCounts = new Array(days.length).fill(0); // 初始化自行车计数数组
        const busCounts = new Array(days.length).fill(0); // 初始化公交车计数数组
        const truckCounts = new Array(days.length).fill(0); // 初始化卡车计数数组

        // 循环处理返回的数据，并将数据填入对应的计数数组
        data.forEach(item => {
          const index = days.indexOf(item.day); // 找到当前数据所在星期的索引
          if (index !== -1) {
            carCounts[index] += item.CarCount;
            bikeCounts[index] += item.BikeCount;
            busCounts[index] += item.BusCount;
            truckCounts[index] += item.TruckCount;
          }
        });

        // 初始化 ECharts 图表
        const chartDom = document.getElementById('weeklyChart');
        const myChart = echarts.init(chartDom);

        const option = {
          title: {
            text: '',
          },
          tooltip: {
            trigger: 'axis'
          },
          legend: {
            data: ['Car Count', 'Bike Count', 'Bus Count', 'Truck Count'] // 设置图例
          },
          xAxis: {
            type: 'category',
            data: days, // 使用完整星期作为 X 轴
            name: '星期',
            boundaryGap: false, // 确保不留空隙，均匀分布
          },
          yAxis: {
            type: 'value',
            name: '车辆计数'
          },
          series: [
            {
              name: 'Car Count',
              type: 'line', // 折线图
              data: carCounts // 车辆计数数据
            },
            {
              name: 'Bike Count',
              type: 'line', // 折线图
              data: bikeCounts // 自行车计数数据
            },
            {
              name: 'Bus Count',
              type: 'line', // 折线图
              data: busCounts // 公交车计数数据
            },
            {
              name: 'Truck Count',
              type: 'line', // 折线图
              data: truckCounts // 卡车计数数据
            }
          ]
        };

        // 使用配置项设置图表
        myChart.setOption(option);

        // 监听窗口尺寸变化，更新图表
        window.addEventListener('resize', function () {
          myChart.resize();
        });
      },

      loadVehicleCounts() {
        axios.get('http://127.0.0.1:5000/api/avg_vehicle_counts')
                .then(response => {
                  const data = response.data;
                  this.renderPlot(data);
                })
                .catch(error => {
                  console.error('Error fetching average vehicle counts:', error);
                });
      },
      renderPlot(data) {
        // 提取时间和车辆计数数据
        const time = data.map(item => item.Time);
        const carCount = data.map(item => item.CarCount);
        const bikeCount = data.map(item => item.BikeCount);
        const busCount = data.map(item => item.BusCount);
        const truckCount = data.map(item => item.TruckCount);

        // 创建图表
        const chartDom = document.getElementById('tu1');
        const myChart = echarts.init(chartDom);
        const option = {
          grid: {
            left: '1%', // 左侧边距设小一些
            right: '1%', // 右侧边距设小一些
            bottom: '3%', // 底部边距
            containLabel: true // 使网格包含坐标轴的标签
          },
          title: {
            text: ''
          },
          tooltip: {
            trigger: 'axis'
          },
          legend: {
            orient: 'horizontal',
            right: '10%',
            top: '10%',
            itemGap: 15,
            itemWidth: 15,
            itemHeight: 15,
            textStyle: {
              fontSize: 14
            },
            data: ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
          },
          xAxis: {
            type: 'category',
            data: time,
            name: 'Time'
          },
          yAxis: {
            type: 'value',
            name: 'Average Count'
          },
          series: [
            {
              name: 'CarCount',
              type: 'line',
              data: carCount
            },
            {
              name: 'BikeCount',
              type: 'line',
              data: bikeCount
            },
            {
              name: 'BusCount',
              type: 'line',
              data: busCount
            },
            {
              name: 'TruckCount',
              type: 'line',
              data: truckCount
            }
          ]
        };

        // 使用配置项设置图表
        myChart.setOption(option);
        // 监听窗口尺寸变化，更新图表
        window.addEventListener('resize', function () {
          myChart.resize();
        });
      }
    },
    mounted(){


      var chartDom = document.getElementById('tu');
      var myChart = echarts.init(chartDom);
      var option;
      option = {
        title: [
        ],
        polar: {
          radius: [10, '85%']
        },
        angleAxis: {
          max:function (value) {
            return value.max+500;
          },
          startAngle: 75
        },
        radiusAxis: {
          type: 'category',
          data: ['重度拥堵', '高度拥堵', '正常','低车流量']
        },
        tooltip: {},
        series: {
          type: 'bar',
          data: [],
          coordinateSystem: 'polar',
          label: {
            show: true,
            position: 'middle',
            formatter: '{b}: {c}'
          },
          itemStyle: {
            normal: {
              color(params) {
                const colorList = ['#5470C6', '#91CC75', '#FAC858', '#EE6666','#73C0DE','#3BA272'];
                return colorList[params.dataIndex];
              }
            }
          }
        }
      };

      this.loadVehicleCounts();
      this.loadVehicleCountsByWeek();  // 初始化调用每周车辆计数数据



      this.request.get("/echarts/members").then(res => {
        // 填空
        // 数据准备完毕之后再set
        option.series.data = res.data;
        //let max1=Math.max.apply(null,res.data);
        myChart.setOption(option)
      }),
              this.request.get("/home/members").then(res => {
                // 填空
                // 数据准备完毕之后再set
                optiondis.series[0].data = res.data
                myChartdis.setOption(optiondis)
              })
    }
  }
</script>
#tu1 {
width: 100%;
height: 600px;
}