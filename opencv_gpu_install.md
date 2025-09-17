# OpenCV GPU 安裝與測試指南

## 1. 安裝對應 CUDA 的 OpenCV GPU Wheel

下載對應自身電腦 **CUDA 預編譯 OpenCV GPU whl 檔** 並安裝，可至以下網址下載：  
👉 [opencv-python-cuda-wheels releases](https://github.com/cudawarped/opencv-python-cuda-wheels/releases)

### 測試成功的環境
- CUDA 12.5  
- cuDNN 9.2.0  

測試下載檔案：
- [opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl](https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.10.0.84/opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl)
- [cuDNN 9.2.0 for CUDA 12 (windows-x86_64)](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.2.0.82_cuda12-archive.zip)

安裝方式：
```bash
pip install opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl
```

---

## 2. 測試 OpenCV GPU 是否安裝成功

執行以下 Python 程式碼：
```python
import cv2
print("OpenCV version:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
```

---

## 注意事項

- **wheel 檔需對應 CUDA、cuDNN 跟 Python 版本**，否則容易報錯。  
- 若遇到 **DLL 缺失、CUDA 報錯**，請再次確認 CUDA、cuDNN 與 OpenCV wheel 版本是否相符。  
- 此方式適合主要利用 Python 介面進行 **GPU 加速影像處理**。  

