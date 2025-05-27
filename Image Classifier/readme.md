
---

# 系統特色

## 1. 多種特徵提取方法

* **顏色直方圖 (Color Histogram)**：支援 BGR、HSV、LAB 色彩空間
* **HOG 特徵 (Histogram of Oriented Gradients)**：可調整方向數、細胞大小等
* **LBP 特徵 (Local Binary Pattern)**：局部二值模式，適合紋理分析
* **梯度特徵 (Gradient)**：Sobel 梯度的統計特徵
* **Haar 特徵 (Haar-like)**：使用濾波器模擬 Haar 小波特徵
* **紋理特徵 (Texture)**：包含 Gabor 濾波器和統計特徵

## 2. 自動參數優化

每種方法都會測試多種參數組合：

* **顏色直方圖**：不同 bin 數量和色彩空間
* **HOG**：不同方向數、像素細胞大小、塊大小
* **LBP**：不同半徑、點數、方法
* **梯度**：不同核大小

## 3. 特徵組合測試

系統會測試 2-3 種特徵的組合，找出最佳的特徵融合方案。

---

## 使用方法

### 準備資料夾結構：

```
your_data_folder/
├── 類別1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 類別2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── 類別3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 修改程式碼中的路徑：

將程式碼中的：

```python
data_folder = "your_data_folder"
```

改為你的實際資料夾路徑。

---

## 執行程式：

系統會自動進行以下步驟：

1. 載入所有圖片
2. 測試每種特徵提取方法的所有參數組合
3. 找出每種方法的最佳參數
4. 測試特徵組合
5. 顯示排序後的結果

---

## 輸出結果：

程式會顯示：

* 每種方法的最佳準確率和參數
* 特徵組合的效果
* 最終排序的結果總結

---

## 需要安裝的套件：

```bash
pip install opencv-python scikit-learn scikit-image matplotlib seaborn numpy
```

---

這個系統會幫你找出最適合你的資料集的特徵提取方法和參數組合，讓你能夠建立一個高準確率的圖片分類器！

---
