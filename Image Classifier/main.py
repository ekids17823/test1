import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from skimage import filters
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ImageFeatureExtractor:
    def __init__(self):
        self.feature_methods = {
            'color_histogram': self.extract_color_histogram,
            'hog': self.extract_hog_features,
            'lbp': self.extract_lbp_features,
            'gradient': self.extract_gradient_features,
            'haar': self.extract_haar_features,
            'texture': self.extract_texture_features
        }
        
    def extract_color_histogram(self, image, bins=32, color_space='BGR'):
        """提取顏色直方圖特徵"""
        if color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        features = []
        for i in range(3):  # 3個顏色通道
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            features.extend(hist.flatten())
        return np.array(features)
    
    def extract_hog_features(self, image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """提取HOG特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=orientations, 
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      visualize=False, feature_vector=True)
        return features
    
    def extract_lbp_features(self, image, radius=3, n_points=24, method='uniform'):
        """提取LBP特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method=method)
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                             range=(0, n_points + 2), density=True)
        return hist
    
    def extract_gradient_features(self, image, ksize=3):
        """提取梯度特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # 統計特徵
        features = [
            np.mean(magnitude), np.std(magnitude),
            np.mean(direction), np.std(direction),
            np.percentile(magnitude, 25), np.percentile(magnitude, 75)
        ]
        return np.array(features)
    
    def extract_haar_features(self, image):
        """提取Haar小波特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 簡化的Haar特徵（使用濾波器模擬）
        # 水平邊緣檢測
        kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        # 垂直邊緣檢測
        kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        # 對角線特徵
        kernel_d1 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        kernel_d2 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
        
        features = []
        for kernel in [kernel_h, kernel_v, kernel_d1, kernel_d2]:
            filtered = cv2.filter2D(gray, -1, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
        
        return np.array(features)
    
    def extract_texture_features(self, image):
        """提取紋理特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # 計算不同方向的紋理特徵
        features = []
        
        # Gabor濾波器
        for theta in [0, 45, 90, 135]:
            real, _ = filters.gabor(gray, frequency=0.6, theta=np.deg2rad(theta))
            features.extend([np.mean(real), np.std(real)])
        
        # 統計特徵
        features.extend([
            np.mean(gray), np.std(gray),
            np.percentile(gray, 25), np.percentile(gray, 75)
        ])
        
        return np.array(features)

class ImageClassifier:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.feature_extractor = ImageFeatureExtractor()
        self.scaler = StandardScaler()
        self.best_results = {}
        
    def load_data(self):
        """載入資料"""
        images = []
        labels = []
        class_names = []
        
        for class_idx, class_name in enumerate(os.listdir(self.data_folder)):
            class_path = os.path.join(self.data_folder, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_names.append(class_name)
            print(f"載入類別: {class_name}")
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 調整圖片大小
                        img = cv2.resize(img, (224, 224))
                        images.append(img)
                        labels.append(class_idx)
        
        print(f"總共載入 {len(images)} 張圖片，{len(class_names)} 個類別")
        return images, labels, class_names
    
    def extract_features(self, images, method, **params):
        """提取特徵"""
        features = []
        for img in images:
            if method == 'color_histogram':
                feat = self.feature_extractor.extract_color_histogram(img, **params)
            elif method == 'hog':
                feat = self.feature_extractor.extract_hog_features(img, **params)
            elif method == 'lbp':
                feat = self.feature_extractor.extract_lbp_features(img, **params)
            elif method == 'gradient':
                feat = self.feature_extractor.extract_gradient_features(img, **params)
            elif method == 'haar':
                feat = self.feature_extractor.extract_haar_features(img)
            elif method == 'texture':
                feat = self.feature_extractor.extract_texture_features(img)
            
            features.append(feat)
        
        return np.array(features)
    
    def combine_features(self, feature_list):
        """組合多種特徵"""
        if len(feature_list) == 1:
            return feature_list[0]
        
        combined = np.concatenate(feature_list, axis=1)
        return combined
    
    def optimize_single_method(self, images, labels, method):
        """優化單一方法的參數"""
        print(f"\n=== 優化 {method} 方法 ===")
        
        # 定義每種方法的參數網格
        param_grids = {
            'color_histogram': {
                'bins': [16, 32, 64],
                'color_space': ['BGR', 'HSV', 'LAB']
            },
            'hog': {
                'orientations': [6, 9, 12],
                'pixels_per_cell': [(8, 8), (16, 16)],
                'cells_per_block': [(2, 2), (3, 3)]
            },
            'lbp': {
                'radius': [1, 2, 3],
                'n_points': [8, 16, 24],
                'method': ['uniform', 'nri_uniform']
            },
            'gradient': {
                'ksize': [3, 5, 7]
            },
            'haar': {},  # 無參數
            'texture': {}  # 無參數
        }
        
        best_accuracy = 0
        best_params = {}
        best_model = None
        
        param_grid = param_grids.get(method, {})
        
        if not param_grid:  # 無參數的方法
            param_combinations = [{}]
        else:
            # 生成所有參數組合
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            param_combinations = []
            
            def generate_combinations(index, current_params):
                if index == len(keys):
                    param_combinations.append(current_params.copy())
                    return
                
                key = keys[index]
                for value in values[index]:
                    current_params[key] = value
                    generate_combinations(index + 1, current_params)
                    del current_params[key]
            
            generate_combinations(0, {})
        
        for params in param_combinations:
            try:
                print(f"測試參數: {params}")
                
                # 提取特徵
                features = self.extract_features(images, method, **params)
                
                # 分割資料
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=42, stratify=labels
                )
                
                # 標準化
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # 訓練模型
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                
                # 預測
                y_pred = rf_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"準確率: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params.copy()
                    best_model = rf_model
                
            except Exception as e:
                print(f"參數組合失敗: {e}")
                continue
        
        self.best_results[method] = {
            'accuracy': best_accuracy,
            'params': best_params,
            'model': best_model
        }
        
        print(f"{method} 最佳準確率: {best_accuracy:.4f}")
        print(f"{method} 最佳參數: {best_params}")
        
        return best_accuracy, best_params
    
    def test_feature_combinations(self, images, labels):
        """測試特徵組合"""
        print("\n=== 測試特徵組合 ===")
        
        methods = list(self.best_results.keys())
        best_combination_accuracy = 0
        best_combination = []
        
        # 測試2-3種特徵的組合
        for r in range(2, min(4, len(methods) + 1)):
            for combination in combinations(methods, r):
                try:
                    print(f"測試組合: {combination}")
                    
                    # 提取各種特徵
                    feature_list = []
                    for method in combination:
                        best_params = self.best_results[method]['params']
                        features = self.extract_features(images, method, **best_params)
                        feature_list.append(features)
                    
                    # 組合特徵
                    combined_features = self.combine_features(feature_list)
                    
                    # 分割資料
                    X_train, X_test, y_train, y_test = train_test_split(
                        combined_features, labels, test_size=0.3, random_state=42, stratify=labels
                    )
                    
                    # 標準化
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 訓練模型
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train_scaled, y_train)
                    
                    # 預測
                    y_pred = rf_model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    print(f"組合準確率: {accuracy:.4f}")
                    
                    if accuracy > best_combination_accuracy:
                        best_combination_accuracy = accuracy
                        best_combination = combination
                        
                except Exception as e:
                    print(f"組合測試失敗: {e}")
                    continue
        
        print(f"\n最佳組合: {best_combination}")
        print(f"最佳組合準確率: {best_combination_accuracy:.4f}")
        
        return best_combination, best_combination_accuracy
    
    def run_full_experiment(self):
        """執行完整實驗"""
        print("開始載入資料...")
        images, labels, class_names = self.load_data()
        
        if len(images) == 0:
            print("未找到圖片，請檢查資料夾路徑")
            return
        
        # 測試所有單一方法
        methods_to_test = ['color_histogram', 'hog', 'lbp', 'gradient', 'haar', 'texture']
        
        for method in methods_to_test:
            try:
                self.optimize_single_method(images, labels, method)
            except Exception as e:
                print(f"方法 {method} 測試失敗: {e}")
        
        # 測試特徵組合
        if len(self.best_results) >= 2:
            best_combination, best_combination_accuracy = self.test_feature_combinations(images, labels)
        
        # 顯示最終結果
        self.display_results()
        
        return self.best_results
    
    def display_results(self):
        """顯示結果"""
        print("\n" + "="*50)
        print("實驗結果總結")
        print("="*50)
        
        # 排序結果
        sorted_results = sorted(self.best_results.items(), 
                              key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (method, result) in enumerate(sorted_results, 1):
            print(f"{i}. {method}")
            print(f"   準確率: {result['accuracy']:.4f}")
            print(f"   最佳參數: {result['params']}")
            print("-" * 30)

# 使用範例
if __name__ == "__main__":
    # 請將 'your_data_folder' 替換為你的資料夾路徑
    data_folder = "your_data_folder"  # 例如: "/path/to/your/image/folder"
    
    # 建立分類器
    classifier = ImageClassifier(data_folder)
    
    # 執行完整實驗
    results = classifier.run_full_experiment()
    
    print("\n實驗完成！")
    print("你可以根據結果選擇最適合的特徵提取方法和參數組合。")