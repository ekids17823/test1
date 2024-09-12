using System;
using System.Timers;
using Renci.SshNet;
using System.Data.SQLite;

class PerformanceMonitor
{
    private static Timer timer;

    public static void Main()
    {
        string dbPath = "Data Source=performance_data.db";

        // Create DB and Table if they don't exist
        using (var conn = new SQLiteConnection(dbPath))
        {
            conn.Open();
            string createTableQuery = @"
                CREATE TABLE IF NOT EXISTS PerformanceData (
                    TimeStamp DATETIME,
                    CPUUsage REAL,
                    RAMUsage REAL,
                    RAMPercentage REAL,
                    GPUMemoryUsage REAL,
                    GPUMemoryPercentage REAL,
                    ReadSpeed REAL,
                    WriteSpeed REAL,
                    TotalDiskSpace REAL,
                    DownloadSpeed REAL,
                    UploadSpeed REAL
                );";
            using (var cmd = new SQLiteCommand(createTableQuery, conn))
            {
                cmd.ExecuteNonQuery();
            }
            // TimeStamp, CPUUsage, RAMUsage, RAMPercentage, TotalDiskSpace, DownloadSpeed, UploadSpeed
        }


        timer = new Timer(30000); // 每30秒執行一次
        timer.Elapsed += CollectData;
        timer.AutoReset = true;
        timer.Enabled = true;

        Console.WriteLine("效能監控開始...按 Enter 鍵停止");
        Console.ReadLine();
    }

    private static void CollectData(Object source, ElapsedEventArgs e)
    {
        string host = "0.0.0.0";
        string username = "user";
        string password = "password";
        using (var client = new SshClient(host, username, password))
        {
            client.Connect();

            // 獲取 CPU 使用率
            var cpuUsage = ExecuteCommand(client, "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'");
            Console.WriteLine(cpuUsage);

            // 獲取 RAM 使用量與使用百分比
            var ramUsage = ExecuteCommand(client, "free -m | awk 'NR==2{printf \"%s\", $3}'");
            var ramPercentage = ExecuteCommand(client, "free | awk 'NR==2{printf \"%.2f\", $3*100/$2 }'");
            Console.WriteLine(ramUsage);
            Console.WriteLine(ramPercentage);

            // 獲取 GPU 記憶體使用量與使用百分比 (如果有 GPU)
            //var gpuMemoryUsage = ExecuteCommand(client, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits");
            //var gpuMemoryPercentage = ExecuteCommand(client, "nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits");
            //Console.WriteLine(gpuMemoryUsage);
            //Console.WriteLine(gpuMemoryPercentage);
            var gpuMemoryUsage = "0";
            var gpuMemoryPercentage = "0";

            // 獲取硬碟空間
            var totalDiskSpace = ExecuteCommand(client, "df -h --total | grep 'total' | awk '{print $2}'");
            Console.WriteLine(totalDiskSpace);

            // 獲取即時讀寫速度 (以 MB/s 表示)
            var readSpeed = ExecuteCommand(client, "iostat -d | awk 'NR==4{print $3}'");
            var writeSpeed = ExecuteCommand(client, "iostat -d | awk 'NR==4{print $4}'");
            Console.WriteLine(readSpeed);
            Console.WriteLine(writeSpeed);

            // 獲取網路速度
            var uploadSpeed = ExecuteCommand(client, "cat /proc/net/dev | grep eth0 | awk '{print $10/1024/1024}'");
            var downloadSpeed = ExecuteCommand(client, "cat /proc/net/dev | grep eth0 | awk '{print $2/1024/1024}'");
            Console.WriteLine(uploadSpeed);
            Console.WriteLine(downloadSpeed);

            client.Disconnect();

            SaveToDatabase(float.Parse(cpuUsage), float.Parse(ramUsage), float.Parse(ramPercentage),
                           float.Parse(gpuMemoryUsage), float.Parse(gpuMemoryPercentage), totalDiskSpace,
                           float.Parse(readSpeed), float.Parse(writeSpeed), float.Parse(uploadSpeed), float.Parse(downloadSpeed));
        }
    }

    private static string ExecuteCommand(SshClient client, string command)
    {
        var cmd = client.CreateCommand(command);
        return cmd.Execute().Trim();
    }

    private static void SaveToDatabase(float cpuUsage, float ramUsage, float ramPercentage, float gpuMemoryUsage, float gpuMemoryPercentage, string totalDiskSpace, float readSpeed, float writeSpeed, float uploadSpeed, float downloadSpeed)
    {
        using (SQLiteConnection conn = new SQLiteConnection("Data Source=performance_data.db;Version=3;"))
        {
            conn.Open();
            string query = "INSERT INTO PerformanceData (TimeStamp, CPUUsage, RAMUsage, RAMPercentage, GPUMemoryUsage, GPUMemoryPercentage, TotalDiskSpace, ReadSpeed, WriteSpeed, UploadSpeed, DownloadSpeed) " +
                           "VALUES (@timestamp, @cpuUsage, @ramUsage, @ramPercentage, @gpuUsage, @gpuPercentage, @totalDisk, @readSpeed, @writeSpeed, @uploadSpeed, @downloadSpeed)";
            using (SQLiteCommand cmd = new SQLiteCommand(query, conn))
            {
                cmd.Parameters.AddWithValue("@timestamp", DateTime.Now);
                cmd.Parameters.AddWithValue("@cpuUsage", cpuUsage);
                cmd.Parameters.AddWithValue("@ramUsage", ramUsage);
                cmd.Parameters.AddWithValue("@ramPercentage", ramPercentage);
                cmd.Parameters.AddWithValue("@gpuUsage", gpuMemoryUsage);
                cmd.Parameters.AddWithValue("@gpuPercentage", gpuMemoryPercentage);
                cmd.Parameters.AddWithValue("@totalDisk", totalDiskSpace);
                cmd.Parameters.AddWithValue("@readSpeed", readSpeed);
                cmd.Parameters.AddWithValue("@writeSpeed", writeSpeed);
                cmd.Parameters.AddWithValue("@uploadSpeed", uploadSpeed);
                cmd.Parameters.AddWithValue("@downloadSpeed", downloadSpeed);
                cmd.ExecuteNonQuery();
            }
            Console.WriteLine("DB Insert Success");
        }
    }
}
