<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Default.aspx.cs" Inherits="WebApplication1.Default" %>

<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title>效能監控資料視覺化</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <form id="form2" runat="server">
        <div>
            <h1>效能監控資料視覺化</h1>

            <label for="startDate">開始日期:</label>
            <input type="date" id="startDate" name="startDate" />

            <label for="endDate">結束日期:</label>
            <input type="date" id="endDate" name="endDate" />

            <button type="submit" runat="server" onserverclick="btnFilter_Click">篩選</button>

            <hr />

            <div style="width:80%">
                <canvas id="cpuChart"></canvas>
            </div>
            <div style="width:80%">
                <canvas id="ramChart"></canvas>
            </div>
            <div style="width:80%">
                <canvas id="diskChart"></canvas>
            </div>
            <div style="width:80%">
                <canvas id="networkChart"></canvas>
            </div>
        </div>
    </form>
    <script type="text/javascript">
        var cpuData = [];
        var ramData = [];
        var diskData = [];
        var networkData = [];
        var labels = [];

        // performanceData 已經在後端被注入
        //console.log(performanceData); // 查看資料

        // 從 JSON 中提取資料
        var labels = performanceData.map(item => convertToDate(item.TimeStamp));
        var cpuData = performanceData.map(item => item.CPUUsage);
        var ramData = performanceData.map(item => item.RAMPercentage);
        //var diskReadSpeed = performanceData.map(item => item.ReadSpeed);
        //var diskWriteSpeed = performanceData.map(item => item.WriteSpeed);
        performanceData.forEach(item => {
            diskData.push({ readSpeed: item.ReadSpeed, writeSpeed: item.WriteSpeed });
        });
        //var networkUploadSpeed = performanceData.map(item => item.UploadSpeed);
        //var networkDownloadSpeed = performanceData.map(item => item.DownloadSpeed);                    
        performanceData.forEach(item => {
            networkData.push({ upload: item.UploadSpeed, download: item.DownloadSpeed });
        });

        //console.log(labels);
        //console.log(cpuData);
        //console.log(ramData);
        //console.log(diskData);
        //console.log(networkData);

        document.addEventListener('DOMContentLoaded', function() {
            var ctxCpu = document.getElementById('cpuChart').getContext('2d');
            new Chart(ctxCpu, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'CPU 使用率 (%)',
                        data: cpuData,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 10, // 限制X軸最多顯示10個刻度
                            }
                        }
                    }
                }
            });

            var ctxRam = document.getElementById('ramChart').getContext('2d');
            new Chart(ctxRam, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'RAM 使用率 (%)',
                        data: ramData,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 10, // 限制X軸最多顯示10個刻度
                            }
                        }
                    }
                }
            });

            var ctxDisk = document.getElementById('diskChart').getContext('2d');
            new Chart(ctxDisk, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '讀取速度 (MB/s)',
                        data: diskData.map(d => d.readSpeed),
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }, {
                        label: '寫入速度 (MB/s)',
                        data: diskData.map(d => d.writeSpeed),
                        borderColor: 'rgba(255, 206, 86, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 10, // 限制X軸最多顯示10個刻度
                            }
                        }
                    }
                }
            });

            var ctxNetwork = document.getElementById('networkChart').getContext('2d');
            new Chart(ctxNetwork, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '上傳速度 (MB/s)',
                        data: networkData.map(n => n.upload),
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 1
                    }, {
                        label: '下載速度 (MB/s)',
                        data: networkData.map(n => n.download),
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 10, // 限制X軸最多顯示10個刻度
                            }
                        }
                    }
                }
            });
        });

        function convertToDate(jsonDate) {
            // 提取數字部分
            var timestamp = jsonDate.match(/\d+/)[0];
    
            // 將 timestamp 轉換為整數，並創建 Date 物件
            var date = new Date(parseInt(timestamp, 10));

            // 提取小時和分鐘，並格式化為兩位數
            var hours = date.getHours().toString().padStart(2, '0');
            var minutes = date.getMinutes().toString().padStart(2, '0');
    
            return hours + ":" + minutes;
        }

    </script>
</body>
</html>
