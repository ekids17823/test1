# OpenCV GPU (CUDA) 安裝指南

本指南說明如何於 Windows 平台下，安裝支援 CUDA 的 OpenCV GPU 預編譯檔，並驗證 CUDA 功能是否可用。

---

## 1. 下載對應自身電腦 CUDA 的 OpenCV GPU whl 檔並安裝

請前往以下網址下載對應您 CUDA 及 Python 版本的 OpenCV CUDA wheel 檔：

[https://github.com/cudawarped/opencv-python-cuda-wheels/releases](https://github.com/cudawarped/opencv-python-cuda-wheels/releases)

### 範例：  
測試成功環境  
- CUDA 12.5  
- cuDNN 9.2.0

#### 下載位置：
- [opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl](https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.10.0.84/opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl)
- [cudnn-windows-x86_64-9.2.0.82_cuda12-archive.zip](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.2.0.82_cuda12-archive.zip)

---

## 2. 安裝步驟

1. 安裝對應 whl 檔，例如（請依照實際 python 路徑、檔名替換）：
   ```sh
   pip install opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl
   ```
2. 解壓下載的 cuDNN zip 檔，將所需的 DLL 複製至對應 CUDA 的 `bin`、`lib` 資料夾。

---

## 3. 驗證 OpenCV GPU 功能是否啟用

於 Python 執行以下程式碼：
```python
import cv2
print("OpenCV version:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
```
若 CUDA 驅動及安裝路徑皆正確，會顯示您的 GPU 裝置數量。

---

## 注意事項

- **whl 檔必須對應 CUDA、cuDNN 及 Python 版本**，否則可能會出現錯誤。
- 若遇到 DLL 缺失、CUDA 報錯，請再確認 CUDA、cuDNN 及 Python 版本是否正確對應及安裝。
- 此安裝方式適合主要利用 Python 介面進行 GPU 加速影像處理。

---

## 參考連結

- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN Downloads](https://developer.nvidia.com/rdp/cudnn-archive)
- [opencv-python-cuda-wheels 專案](https://github.com/cudawarped/opencv-python-cuda-wheels)

---