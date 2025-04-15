# 測試圖片與遮罩自動產生腳本
import os
import numpy as np
from PIL import Image

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def gen_good_bad_example():
    base = '.'
    good_img_dir = os.path.join(base, 'Good', 'IMG')
    good_mask_dir = os.path.join(base, 'Good', 'Mask')
    bad_img_dir = os.path.join(base, 'Bad', 'IMG')
    bad_mask_dir = os.path.join(base, 'Bad', 'Mask')
    for d in [good_img_dir, good_mask_dir, bad_img_dir, bad_mask_dir]:
        make_dir(d)
    # Good: 全白遮罩
    for i in range(10):
        arr = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.ones((256, 256), dtype=np.uint8) * 255
        Image.fromarray(arr).save(os.path.join(good_img_dir, f'good_{i+1:03d}.png'))
        Image.fromarray(mask).save(os.path.join(good_mask_dir, f'good_{i+1:03d}.png'))
    # Bad: 遮罩有黑塊
    for i in range(10):
        arr = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.ones((256, 256), dtype=np.uint8) * 255
        cv = np.random.randint(0, 128, (80, 80), dtype=np.uint8)
        x, y = np.random.randint(0, 176, 2)
        mask[x:x+80, y:y+80] = cv
        Image.fromarray(arr).save(os.path.join(bad_img_dir, f'bad_{i+1:03d}.png'))
        Image.fromarray(mask).save(os.path.join(bad_mask_dir, f'bad_{i+1:03d}.png'))

if __name__ == '__main__':
    gen_good_bad_example()
