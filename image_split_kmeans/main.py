import os
import yaml
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.models as models
import math

# 讀取config.yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

GRID = config['split']['grid']
OVERLAP = config['split']['overlap']
INPUTS = config['input']
OUTPUTS = config['output']
KMEANS_N = config['kmeans']['n_clusters']
PCA_DIM = config['kmeans']['pca_dim']

os.makedirs(OUTPUTS['good_dir'], exist_ok=True)
os.makedirs(OUTPUTS['bad_dir'], exist_ok=True)

def get_image_mask_pairs(img_dir, mask_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    pairs = []
    for img in img_files:
        mask = img.replace('.jpg', '.png').replace('.jpeg', '.png')
        if not os.path.exists(os.path.join(mask_dir, mask)):
            continue
        pairs.append((os.path.join(img_dir, img), os.path.join(mask_dir, mask)))
    return pairs

def sliding_window(img, grid, overlap):
    h, w = img.shape[:2]
    n_rows, n_cols = grid
    win_h = h // n_rows
    win_w = w // n_cols
    step_h = int(win_h * (1 - overlap))
    step_w = int(win_w * (1 - overlap))
    windows = []
    for r in range(0, h - win_h + 1, step_h):
        for c in range(0, w - win_w + 1, step_w):
            windows.append((r, c, img[r:r+win_h, c:c+win_w]))
    return windows

def is_mask_bad(mask_patch):
    # 全白為good，有黑色為bad
    if np.all(mask_patch >= 250):
        return False
    return True

def save_patch(patch, outdir, basename, idx):
    fname = f"{os.path.splitext(basename)[0]}_{idx}.png"
    Image.fromarray(patch).save(os.path.join(outdir, fname))
    return os.path.join(outdir, fname)

def process_set(img_dir, mask_dir, out_good, out_bad):
    pairs = get_image_mask_pairs(img_dir, mask_dir)
    all_patches, all_labels = [], []
    for img_path, mask_path in pairs:
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        img_patches = sliding_window(img, GRID, OVERLAP)
        mask_patches = sliding_window(mask, GRID, OVERLAP)
        for i, ((r, c, patch), (_, _, mask_patch)) in enumerate(zip(img_patches, mask_patches)):
            if patch.shape[0] != img.shape[0] // GRID[0] or patch.shape[1] != img.shape[1] // GRID[1]:
                continue  # 跳過不齊全的patch
            label = 'bad' if is_mask_bad(mask_patch) else 'good'
            outdir = out_bad if label == 'bad' else out_good
            save_patch(patch, outdir, os.path.basename(img_path), i)
            all_patches.append(cv2.resize(patch, (64, 64)).flatten())  # 統一特徵維度
            all_labels.append(label)
    return np.array(all_patches), np.array(all_labels)

def extract_cnn_features(patches, model_name='resnet18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        feat_dim = 4096
        model = torch.nn.Sequential(*list(model.features), torch.nn.AdaptiveAvgPool2d((7,7)), torch.nn.Flatten(), *list(model.classifier)[:-1])
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        feat_dim = 1280
        model = torch.nn.Sequential(*(list(model.features)), torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten())
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        feat_dim = 1024
        model = torch.nn.Sequential(*(list(model.features)), torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten())
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = 1280
        model = torch.nn.Sequential(*(list(model.features)), torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten())
    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        feat_dim = 512
        model = torch.nn.Sequential(*(list(model.features)), torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten())
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model.eval()
    model.to(device)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for patch in patches:
            img = patch.reshape(64, 64, 3).astype('uint8')
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).squeeze().cpu().numpy().reshape(-1)[:feat_dim]
            feats.append(feat)
    return np.array(feats)

def main():
    # Good/Bad都要處理
    patches1, labels1 = process_set(INPUTS['good_img_dir'], INPUTS['good_mask_dir'], OUTPUTS['good_dir'], OUTPUTS['bad_dir'])
    patches2, labels2 = process_set(INPUTS['bad_img_dir'], INPUTS['bad_mask_dir'], OUTPUTS['good_dir'], OUTPUTS['bad_dir'])
    X = np.concatenate([patches1, patches2], axis=0)
    y = np.concatenate([labels1, labels2], axis=0)
    # KMeans+PCA (flatten 特徵)
    pca = PCA(n_components=PCA_DIM)
    X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=KMEANS_N, random_state=0)
    y_pred = kmeans.fit_predict(X_pca)
    cnn_models = ['resnet18', 'resnet50', 'vgg16', 'mobilenet_v2', 'densenet121', 'efficientnet_b0', 'squeezenet1_1']
    cnn_features = {}
    pca_cnn = {}
    kmeans_cnn = {}
    y_pred_cnn = {}
    for m in cnn_models:
        print(f'Extracting CNN features with {m}...')
        feats = extract_cnn_features(X, model_name=m)
        cnn_features[m] = feats
        pca_cnn[m] = PCA(n_components=PCA_DIM).fit_transform(feats)
        kmeans_cnn[m] = KMeans(n_clusters=KMEANS_N, random_state=0).fit(pca_cnn[m])
        y_pred_cnn[m] = kmeans_cnn[m].labels_
    # 畫圖
    n_models = len(cnn_models)
    ncols = 4
    nrows = math.ceil((n_models+1)/ncols)
    plt.figure(figsize=(5*ncols, 5*nrows))
    plt.subplot(nrows, ncols, 1)
    for label, color in [('good','g'),('bad','r')]:
        idx = (y==label)
        plt.scatter(X_pca[idx,0], X_pca[idx,1], c=color, label=label, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='blue', marker='x', s=100, label='Centers')
    plt.legend()
    plt.title('Flatten+PCA')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    for i, m in enumerate(cnn_models):
        plt.subplot(nrows, ncols, i+2)
        for label, color in [('good','g'),('bad','r')]:
            idx = (y==label)
            plt.scatter(pca_cnn[m][idx,0], pca_cnn[m][idx,1], c=color, label=label, alpha=0.5)
        plt.scatter(kmeans_cnn[m].cluster_centers_[:,0], kmeans_cnn[m].cluster_centers_[:,1], c='blue', marker='x', s=100, label='Centers')
        plt.title(f'{m}+PCA')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
