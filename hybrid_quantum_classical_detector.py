import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from improved_dataset import load_deepfake_dataset

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HybridQuantumClassicalDetector(nn.Module):
    """
    量子-经典混合深伪检测器
    结合经典神经网络的特征提取能力和量子计算的优势
    """
    
    def __init__(self, input_dim=6, n_qubits=4, n_layers=3, hidden_dim=16):
        super(HybridQuantumClassicalDetector, self).__init__()
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # 经典预处理网络
        self.classical_preprocessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_qubits)  # 输出维度匹配量子比特数
        )
        
        # 量子设备和电路
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.quantum_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3, dtype=torch.float32) * 0.1)
        
        # 经典后处理网络
        self.classical_postprocessor = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 创建量子电路
        self._create_quantum_circuit()
        
        self.training_history = {'loss': [], 'accuracy': []}
    
    def _create_quantum_circuit(self):
        """创建量子电路"""
        
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, params):
            # 数据编码层
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 变分量子层
            for layer in range(self.n_layers):
                # 参数化旋转门
                for i in range(self.n_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                
                # 纠缠层
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # 测量所有量子比特
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
    
    def forward(self, x):
        """前向传播"""
        # 确保输入是float32类型
        x = x.float()
        
        # 经典预处理
        classical_features = self.classical_preprocessor(x)
        
        # 量子处理
        quantum_outputs = []
        for i in range(x.shape[0]):
            quantum_out = self.quantum_circuit(classical_features[i], self.quantum_params)
            # 确保量子输出是float32类型
            quantum_out_tensor = torch.stack(quantum_out).float()
            quantum_outputs.append(quantum_out_tensor)
        
        quantum_features = torch.stack(quantum_outputs)
        
        # 经典后处理
        output = self.classical_postprocessor(quantum_features)
        
        return output.squeeze()
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=200, batch_size=32, learning_rate=0.001):
        """训练混合模型"""
        print("开始训练量子-经典混合深伪检测器...")
        
        # 转换为PyTorch张量，确保是float32类型
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # 训练循环
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            # 批量训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 记录训练历史
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / (len(X_train) // batch_size + 1)
                
                # 计算训练准确率
                self.eval()
                with torch.no_grad():
                    train_outputs = self.forward(X_train_tensor)
                    train_preds = (train_outputs > 0.5).float()
                    train_acc = (train_preds == y_train_tensor).float().mean().item()
                
                self.training_history['loss'].append(avg_loss)
                self.training_history['accuracy'].append(train_acc)
                
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  训练损失: {avg_loss:.4f}")
                print(f"  训练准确率: {train_acc:.4f}")
                
                # 验证集评估
                if X_val is not None:
                    with torch.no_grad():
                        val_outputs = self.forward(X_val_tensor)
                        val_preds = (val_outputs > 0.5).float()
                        val_acc = (val_preds == y_val_tensor).float().mean().item()
                        print(f"  验证准确率: {val_acc:.4f}")
                
                print()
    
    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
            predictions = (outputs > 0.5).float().numpy()
        return predictions.astype(int)
    
    def predict_proba(self, X):
        """预测概率"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
        return outputs.numpy()
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("评估混合模型性能...")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"测试准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        if not self.training_history['loss']:
            print("没有训练历史可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.training_history['loss'])
        ax1.set_title('混合模型训练损失')
        ax1.set_xlabel('Epoch (×20)')
        ax1.set_ylabel('损失')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.training_history['accuracy'])
        ax2.set_title('混合模型训练准确率')
        ax2.set_xlabel('Epoch (×20)')
        ax2.set_ylabel('准确率')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例和对比实验
if __name__ == "__main__":
    # 加载数据
    print("加载数据集...")
    X_train, X_test, y_train, y_test, scaler, pca = load_deepfake_dataset(
        sample_size=1000,
        use_pca=True,
        n_components=6
    )
    
    # 创建验证集
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"训练集: {X_train_split.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 创建并训练混合模型
    hybrid_model = HybridQuantumClassicalDetector(
        input_dim=6,
        n_qubits=4,
        n_layers=3,
        hidden_dim=16
    )
    
    # 训练模型
    hybrid_model.train_model(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.001
    )
    
    # 评估模型
    results = hybrid_model.evaluate(X_test, y_test)
    
    # 绘制训练历史
    hybrid_model.plot_training_history()
    
    print("\n混合模型训练完成！")