import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, hog
from skimage import color
import warnings
warnings.filterwarnings('ignore')

class ImageClassifier:
    def __init__(self, data_dir):
        """
        初始化圖片分類器
        data_dir: 包含三個子資料夾的主資料夾路徑
        """
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.class_names = []
        self.scaler = StandardScaler()
        
    def load_data(self):
        """載入圖片和標籤"""
        print("正在載入圖片資料...")
        
        # 獲取類別名稱
        self.class_names = [d for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            print(f"載入類別 {class_name}...")
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        # 調整圖片大小為統一尺寸
                        img = cv2.resize(img, (128, 128))
                        self.images.append(img)
                        self.labels.append(class_idx)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"總共載入 {len(self.images)} 張圖片，{len(self.class_names)} 個類別")
    
    def extract_color_histogram(self, image):
        """提取顏色直方圖特徵"""
        # 轉換到HSV色彩空間
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 計算各通道的直方圖
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
        
        # 正規化並組合
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        return hist / (hist.sum() + 1e-7)
    
    def extract_hog_features(self, image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """提取HOG特徵[1]"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, 
                      orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      block_norm='L2-Hys',
                      visualize=False)
        return features
    
    def extract_lbp_features(self, image, radius=3, n_points=24):
        """提取LBP特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # 計算LBP直方圖
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                              range=(0, n_points + 2), density=True)
        return hist
    
    def extract_gradient_features(self, image):
        """提取梯度特徵"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 計算x和y方向的梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 計算梯度幅值和方向
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # 統計特徵
        features = [
            np.mean(magnitude), np.std(magnitude),
            np.mean(direction), np.std(direction),
            np.percentile(magnitude, 25), np.percentile(magnitude, 75)
        ]
        return np.array(features)
    
    def extract_combined_features(self, image, feature_types=['color', 'hog', 'lbp']):
        """提取組合特徵"""
        features = []
        
        if 'color' in feature_types:
            color_feat = self.extract_color_histogram(image)
            features.append(color_feat)
        
        if 'hog' in feature_types:
            hog_feat = self.extract_hog_features(image)
            features.append(hog_feat)
        
        if 'lbp' in feature_types:
            lbp_feat = self.extract_lbp_features(image)
            features.append(lbp_feat)
            
        if 'gradient' in feature_types:
            grad_feat = self.extract_gradient_features(image)
            features.append(grad_feat)
        
        return np.concatenate(features)
    
    def prepare_features(self, feature_types=['color', 'hog', 'lbp']):
        """準備特徵向量"""
        print(f"提取特徵: {feature_types}")
        features = []
        
        for img in self.images:
            feat = self.extract_combined_features(img, feature_types)
            features.append(feat)
        
        features = np.array(features)
        print(f"特徵維度: {features.shape}")
        return features
    
    def optimize_svm(self, X_train, y_train, X_test, y_test):
        """優化SVM參數"""
        print("優化SVM參數...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        train_acc = best_svm.score(X_train, y_train)
        test_acc = best_svm.score(X_test, y_test)
        
        print(f"最佳SVM參數: {grid_search.best_params_}")
        print(f"SVM訓練準確率: {train_acc:.4f}")
        print(f"SVM測試準確率: {test_acc:.4f}")
        
        return best_svm, test_acc, grid_search.best_params_
    
    def optimize_knn(self, X_train, y_train, X_test, y_test):
        """優化k-NN參數"""
        print("優化k-NN參數...")
        
        param_grid = {
            'n_neighbors': np.arange(1, 31, 2),
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_knn = grid_search.best_estimator_
        train_acc = best_knn.score(X_train, y_train)
        test_acc = best_knn.score(X_test, y_test)
        
        print(f"最佳k-NN參數: {grid_search.best_params_}")
        print(f"k-NN訓練準確率: {train_acc:.4f}")
        print(f"k-NN測試準確率: {test_acc:.4f}")
        
        return best_knn, test_acc, grid_search.best_params_
    
    def optimize_random_forest(self, X_train, y_train, X_test, y_test):
        """優化隨機森林參數"""
        print("優化隨機森林參數...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier()
        random_search = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, 
                                         scoring='accuracy', n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        best_rf = random_search.best_estimator_
        train_acc = best_rf.score(X_train, y_train)
        test_acc = best_rf.score(X_test, y_test)
        
        print(f"最佳隨機森林參數: {random_search.best_params_}")
        print(f"隨機森林訓練準確率: {train_acc:.4f}")
        print(f"隨機森林測試準確率: {test_acc:.4f}")
        
        return best_rf, test_acc, random_search.best_params_
    
    def run_experiments(self):
        """執行所有實驗"""
        if len(self.images) == 0:
            self.load_data()
        
        # 不同的特徵組合
        feature_combinations = [
            ['color'],
            ['hog'],
            ['lbp'],
            ['gradient'],
            ['color', 'hog'],
            ['color', 'lbp'],
            ['hog', 'lbp'],
            ['color', 'hog', 'lbp'],
            ['color', 'hog', 'lbp', 'gradient']
        ]
        
        results = []
        
        for features in feature_combinations:
            print(f"\n{'='*50}")
            print(f"實驗: {' + '.join(features)} 特徵")
            print(f"{'='*50}")
            
            # 提取特徵
            X = self.prepare_features(features)
            
            # 標準化特徵
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割資料
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            
            # 測試不同分類器
            exp_results = {
                'features': features,
                'classifiers': {}
            }
            
            # SVM
            svm_model, svm_acc, svm_params = self.optimize_svm(X_train, y_train, X_test, y_test)
            exp_results['classifiers']['SVM'] = {
                'accuracy': svm_acc,
                'params': svm_params
            }
            
            # k-NN
            knn_model, knn_acc, knn_params = self.optimize_knn(X_train, y_train, X_test, y_test)
            exp_results['classifiers']['k-NN'] = {
                'accuracy': knn_acc,
                'params': knn_params
            }
            
            # 隨機森林
            rf_model, rf_acc, rf_params = self.optimize_random_forest(X_train, y_train, X_test, y_test)
            exp_results['classifiers']['Random Forest'] = {
                'accuracy': rf_acc,
                'params': rf_params
            }
            
            results.append(exp_results)
            
            # 找出最佳分類器
            best_classifier = max(exp_results['classifiers'].items(), 
                                key=lambda x: x[1]['accuracy'])
            print(f"\n最佳分類器: {best_classifier[0]} (準確率: {best_classifier[1]['accuracy']:.4f})")
        
        # 總結所有實驗結果
        self.summarize_results(results)
        
        return results
    
    def summarize_results(self, results):
        """總結實驗結果"""
        print(f"\n{'='*80}")
        print("實驗結果總結")
        print(f"{'='*80}")
        
        best_overall = None
        best_accuracy = 0
        
        for result in results:
            features_str = ' + '.join(result['features'])
            print(f"\n特徵組合: {features_str}")
            print("-" * 40)
            
            for classifier, metrics in result['classifiers'].items():
                accuracy = metrics['accuracy']
                print(f"{classifier:15}: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_overall = {
                        'features': result['features'],
                        'classifier': classifier,
                        'accuracy': accuracy,
                        'params': metrics['params']
                    }
        
        print(f"\n{'='*80}")
        print("最佳配置")
        print(f"{'='*80}")
        print(f"特徵組合: {' + '.join(best_overall['features'])}")
        print(f"分類器: {best_overall['classifier']}")
        print(f"準確率: {best_overall['accuracy']:.4f}")
        print(f"最佳參數: {best_overall['params']}")

# 使用範例
if __name__ == "__main__":
    # 設定資料夾路徑（請修改為您的資料夾路徑）
    data_directory = "dataset"
    
    # 創建分類器實例
    classifier = ImageClassifier(data_directory)
    
    # 執行實驗
    results = classifier.run_experiments()
