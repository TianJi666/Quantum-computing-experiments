import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_deepfake_dataset(
    real_dir: str = './real_png',
    fake_dir: str = './fake_png',
    sample_size: int = 1000,  # 增加样本数
    sample_ratio: float = 1.0,  # 添加这个参数
    image_size=(16, 16),     # 增加图像尺寸
    use_pca=True,
    n_components=8,          # 增加PCA维度
    random_state: int = 42,  # 添加这个参数
    test_size: float = 0.2   # 添加这个参数
):
    """
    加载深伪检测数据集
    
    参数:
    - real_dir: 真实图像文件夹
    - fake_dir: 伪造图像文件夹
    - sample_ratio: 假图:真图比例
    - sample_size: 每类样本数量
    - image_size: 图像尺寸
    - test_size: 测试集比例
    - use_pca: 是否使用PCA降维
    - n_components: PCA降维后的维度
    
    返回:
    - X_train, X_test, y_train, y_test: 训练和测试数据
    - scaler, pca: 预处理器（如果使用）
    """
    
    # 1. 收集文件路径
    real_files = [os.path.join(real_dir, f) 
                  for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fake_files = [os.path.join(fake_dir, f) 
                  for f in os.listdir(fake_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到真实图像: {len(real_files)} 张")
    print(f"找到伪造图像: {len(fake_files)} 张")
    
    # 2. 确定采样数量
    n_real = min(sample_size, len(real_files))
    n_fake = min(int(sample_ratio * sample_size), len(fake_files))
    
    # 3. 随机采样
    random.seed(random_state)
    real_selected = random.sample(real_files, n_real)
    fake_selected = random.sample(fake_files, n_fake)
    
    print(f"采样真实图像: {n_real} 张")
    print(f"采样伪造图像: {n_fake} 张")
    
    # 4. 加载和预处理图像
    def load_images(file_list, label):
        X, y = [], []
        for i, path in enumerate(file_list):
            try:
                # 加载图像并转换为灰度
                img = Image.open(path).convert('L').resize(image_size)
                # 归一化到[0,1]
                arr = np.array(img, dtype=np.float32) / 255.0
                # 展平
                X.append(arr.flatten())
                y.append(label)
                
                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1}/{len(file_list)} 张图像")
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
                continue
        return np.array(X), np.array(y)
    
    print("\n加载真实图像...")
    X_real, y_real = load_images(real_selected, 0)  # 真实图像标签为0
    
    print("\n加载伪造图像...")
    X_fake, y_fake = load_images(fake_selected, 1)  # 伪造图像标签为1
    
    # 5. 合并数据
    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])
    
    print(f"\n总数据形状: {X.shape}")
    print(f"标签分布: 真实={np.sum(y==0)}, 伪造={np.sum(y==1)}")
    
    # 6. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. PCA降维（可选）
    pca = None
    if use_pca:
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"PCA降维后形状: {X_scaled.shape}")
        print(f"解释方差比: {pca.explained_variance_ratio_}")
        print(f"累计解释方差: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    # 8. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"\n训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"训练集标签分布: 真实={np.sum(y_train==0)}, 伪造={np.sum(y_train==1)}")
    print(f"测试集标签分布: 真实={np.sum(y_test==0)}, 伪造={np.sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test, scaler, pca

# 使用示例
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, pca = load_deepfake_dataset(
        sample_size=500,  # 从200改为500
        use_pca=True,
        n_components=6
    )
    
    # 改进的特征提取
    def extract_features(img_array):
        """提取更丰富的特征"""
        features = []
        
        # 基本统计特征
        features.extend([
            np.mean(img_array),
            np.std(img_array),
            np.min(img_array),
            np.max(img_array)
        ])
        
        # 纹理特征（简化版）
        grad_x = np.gradient(img_array, axis=0)
        grad_y = np.gradient(img_array, axis=1)
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])
        
        return features