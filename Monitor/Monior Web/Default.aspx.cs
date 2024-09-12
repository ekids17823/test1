using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SQLite;
using System.Web.Script.Serialization;
using System.Web.UI;

namespace WebApplication1
{
    public partial class Default : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            if (!IsPostBack)
            {
                LoadPerformanceData();
            }
        }

        protected void LoadPerformanceData()
        {
            string startDate = Request.Form["startDate"];
            string endDate = Request.Form["endDate"];

            using (SQLiteConnection conn = new SQLiteConnection("Data Source=" + Server.MapPath("~/App_Data/performance_data.db") + ";Version=3;"))
            {
                conn.Open();
                string query = "SELECT * FROM PerformanceData WHERE TimeStamp BETWEEN @startDate AND @endDate ORDER BY TimeStamp ASC";
                using (SQLiteCommand cmd = new SQLiteCommand(query, conn))
                {
                    cmd.Parameters.AddWithValue("@startDate", startDate ?? DateTime.Now.ToString("yyyy-MM-dd"));
                    cmd.Parameters.AddWithValue("@endDate", endDate ?? DateTime.Now.AddDays(1).ToString("yyyy-MM-dd"));

                    SQLiteDataAdapter da = new SQLiteDataAdapter(cmd);
                    DataTable dt = new DataTable();
                    da.Fill(dt);

                    //rptChartData.DataSource = dt;
                    //rptChartData.DataBind();

                    // 將 DataTable 轉換為 JSON 格式
                    JavaScriptSerializer js = new JavaScriptSerializer();
                    string jsonData = js.Serialize(ConvertDataTableToDictionary(dt));

                    // 將 JSON 資料傳遞到前端
                    ClientScript.RegisterStartupScript(this.GetType(), "setData", $"var performanceData = {jsonData};", true);
                }
            }
        }

        protected void btnFilter_Click(object sender, EventArgs e)
        {
            LoadPerformanceData();
        }

        // 將 DataTable 轉換為 Dictionary 以便序列化成 JSON
        private static List<Dictionary<string, object>> ConvertDataTableToDictionary(DataTable dt)
        {
            var columns = dt.Columns;
            var rows = new List<Dictionary<string, object>>();
            foreach (DataRow row in dt.Rows)
            {
                var rowDict = new Dictionary<string, object>();
                foreach (DataColumn col in columns)
                {
                    rowDict[col.ColumnName] = row[col];
                }
                rows.Add(rowDict);
            }
            return rows;
        }

    }
}
