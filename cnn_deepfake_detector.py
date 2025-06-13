import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from improved_dataset import load_deepfake_dataset
import time

class DeepfakeCNN(nn.Module):
    """
    专门用于深伪检测的CNN模型
    适用于16x16灰度图像输入
    """
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.5):
        super(DeepfakeCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 4x4 -> 2x2
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # 2x2 -> 1x1
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # 第三个卷积块
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # 第四个卷积块
        x = self.adaptive_pool(self.relu(self.bn4(self.conv4(x))))
        
        # 全连接层
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class DeepfakeDetector:
    """
    深伪检测器主类
    """
    def __init__(self, model_params=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 默认模型参数
        default_params = {
            'input_channels': 1,
            'num_classes': 2,
            'dropout_rate': 0.5
        }
        if model_params:
            default_params.update(model_params)
            
        self.model = DeepfakeCNN(**default_params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
    def prepare_data(self, X, y, image_size=(16, 16)):
        """
        准备数据用于CNN训练
        """
        # 如果数据是展平的，重塑为图像格式
        if len(X.shape) == 2:
            # 假设数据已经是展平的图像数据
            expected_pixels = image_size[0] * image_size[1]
            if X.shape[1] == expected_pixels:
                X = X.reshape(-1, 1, image_size[0], image_size[1])
            else:
                raise ValueError(f"数据维度不匹配。期望 {expected_pixels} 个像素，得到 {X.shape[1]}")
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, learning_rate=0.001, 
              weight_decay=1e-4, patience=10):
        """
        训练模型
        """
        print("开始训练CNN深伪检测模型...")
        
        # 准备数据
        X_train_tensor, y_train_tensor = self.prepare_data(X_train, y_train)
        X_val_tensor, y_val_tensor = self.prepare_data(X_val, y_val)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 设置优化器和学习率调度器
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史记录
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                
                val_losses.append(val_loss.item())
                val_accuracies.append(val_accuracy)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_deepfake_cnn.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'  训练损失: {avg_train_loss:.4f}')
                print(f'  验证损失: {val_loss:.4f}')
                print(f'  验证准确率: {val_accuracy:.4f}')
                print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
                print()
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_deepfake_cnn.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'training_time': training_time,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X):
        """
        预测
        """
        self.model.eval()
        X_tensor, _ = self.prepare_data(X, np.zeros(len(X)))  # 虚拟标签
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy(), probabilities.cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        """
        predictions, probabilities = self.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print("模型评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        
        return results
    
    def plot_training_history(self, history):
        """
        绘制训练历史
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['train_losses'], label='训练损失', color='blue')
        ax1.plot(history['val_losses'], label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['val_accuracies'], label='验证准确率', color='green')
        ax2.set_title('验证准确率')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        绘制混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['真实', '伪造'],
                   yticklabels=['真实', '伪造'])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    主函数：完整的深伪检测流程
    """
    print("=" * 60)
    print("CNN深伪检测系统")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据集...")
    X_train, X_test, y_train, y_test, scaler, pca = load_deepfake_dataset(
        sample_size=500,
        image_size=(16, 16),
        use_pca=False,  # 对于CNN，我们不使用PCA
        random_state=42
    )
    
    # 2. 创建验证集
    val_split = int(0.8 * len(X_train))
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 3. 创建检测器
    print("\n2. 创建CNN检测器...")
    detector = DeepfakeDetector()
    
    # 4. 训练模型
    print("\n3. 训练模型...")
    history = detector.train(
        X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=10
    )
    
    # 5. 评估模型
    print("\n4. 评估模型...")
    results = detector.evaluate(X_test, y_test)
    
    # 6. 可视化结果
    print("\n5. 生成可视化结果...")
    detector.plot_training_history(history)
    detector.plot_confusion_matrix(y_test, results['predictions'])
    
    # 7. 保存模型
    torch.save(detector.model.state_dict(), 'final_deepfake_cnn.pth')
    print("\n模型已保存为 'final_deepfake_cnn.pth'")
    
    print("\n=" * 60)
    print("深伪检测完成！")
    print("=" * 60)
    
    return detector, results, history

if __name__ == "__main__":
    detector, results, history = main()