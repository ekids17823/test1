# Image Split & KMeans Example

本專案可將 Good/Bad 圖片依設定切割，並根據遮罩分類，最後用KMeans聚類分析分布。

## 使用方式
1. 安裝依賴：`pip install -r requirements.txt`
2. 調整 `config.yaml` 參數（切割數、重疊率等）
3. 準備圖片與遮罩，放入 Good/IMG, Good/Mask, Bad/IMG, Bad/Mask
4. 執行主程式：`python main.py`
5. 輸出結果於 `output/Good`、`output/Bad`，並產生分布圖 scatter.png

## config.yaml 說明
- split.grid: 切割為幾x幾
- split.overlap: 每塊重疊比例 (0~1)
- input: 圖片與遮罩資料夾
- output: 切割後分類資料夾
- kmeans: 聚類數與降維維度

## 注意
- 遮罩全白視為Good，有黑色視為Bad
- 支援多種切割與重疊率自訂
