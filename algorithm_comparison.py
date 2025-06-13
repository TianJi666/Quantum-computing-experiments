import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from improved_dataset import load_deepfake_dataset
from hybrid_quantum_classical_detector import HybridQuantumClassicalDetector
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_classical_methods(X_train, X_test, y_train, y_test):
    """
    运行经典机器学习方法
    """
    print("运行经典机器学习方法...")
    
    methods = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in methods.items():
        print(f"训练 {name}...")
        start_time = time.time()
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time
        }
        
        print(f"{name} - 准确率: {accuracy:.4f}, 训练时间: {training_time:.2f}秒")
    
    return results

def run_hybrid_quantum_model(X_train, X_test, y_train, y_test):
    """
    运行量子经典混合模型
    """
    print("\n运行量子经典混合模型...")
    
    # 创建验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 创建混合模型
    hybrid_model = HybridQuantumClassicalDetector(
        input_dim=6,
        n_qubits=4,
        n_layers=3,
        hidden_dim=16
    )
    
    start_time = time.time()
    
    # 训练模型（减少epochs以加快对比）
    hybrid_model.train_model(
        X_train_split, y_train_split,
        X_val, y_val,
        epochs=100,  # 减少训练轮数以加快对比
        batch_size=32,
        learning_rate=0.001
    )
    
    training_time = time.time() - start_time
    
    # 评估模型
    y_pred = hybrid_model.predict(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"量子混合模型 - 准确率: {accuracy:.4f}, 训练时间: {training_time:.2f}秒")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time
    }, hybrid_model

def plot_comparison(classical_results, hybrid_result):
    """
    绘制算法对比图表
    """
    # 准备数据
    all_results = classical_results.copy()
    all_results['量子混合模型'] = hybrid_result
    
    algorithms = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('量子经典混合算法 vs 经典算法性能对比', fontsize=16, fontweight='bold')
    
    # 1. 性能指标对比（柱状图）
    ax1 = axes[0, 0]
    x = np.arange(len(algorithms))
    width = 0.2
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [all_results[alg][metric] for alg in algorithms]
        ax1.bar(x + i*width, values, width, label=metric_name, alpha=0.8)
    
    ax1.set_xlabel('算法')
    ax1.set_ylabel('性能分数')
    ax1.set_title('性能指标对比')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 2. 训练时间对比
    ax2 = axes[0, 1]
    training_times = [all_results[alg]['training_time'] for alg in algorithms]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    bars = ax2.bar(algorithms, training_times, color=colors, alpha=0.8)
    ax2.set_xlabel('算法')
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('训练时间对比')
    ax2.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # 3. 准确率雷达图
    ax3 = axes[0, 2]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    
    for i, alg in enumerate(algorithms):
        values = [all_results[alg][metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        ax3.plot(angles, values, 'o-', linewidth=2, label=alg)
        ax3.fill(angles, values, alpha=0.1)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metric_names)
    ax3.set_ylim(0, 1)
    ax3.set_title('性能雷达图', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 4. 算法特点对比表
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    # 创建特点对比表
    features = [
        ['算法类型', 'SVM', 'Random Forest', 'MLP', '量子混合'],
        ['计算复杂度', '中等', '低', '高', '高'],
        ['可解释性', '中等', '高', '低', '中等'],
        ['量子优势', '无', '无', '无', '有'],
        ['适用数据量', '中小', '大', '大', '中小']
    ]
    
    table = ax4.table(cellText=features, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(features)):
        for j in range(len(features[0])):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  # 第一列
                cell.set_facecolor('#E8F5E8')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F5F5F5')
    
    ax4.set_title('算法特点对比', fontweight='bold', pad=20)
    
    # 5. 性能排名
    ax5 = axes[1, 1]
    
    # 计算综合得分（准确率权重0.4，其他各0.2）
    scores = {}
    for alg in algorithms:
        score = (all_results[alg]['accuracy'] * 0.4 + 
                all_results[alg]['precision'] * 0.2 + 
                all_results[alg]['recall'] * 0.2 + 
                all_results[alg]['f1_score'] * 0.2)
        scores[alg] = score
    
    # 排序
    sorted_algs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    alg_names = [item[0] for item in sorted_algs]
    alg_scores = [item[1] for item in sorted_algs]
    
    colors_rank = ['gold', 'silver', '#CD7F32', 'lightblue']  # 金银铜色
    bars = ax5.barh(alg_names, alg_scores, color=colors_rank[:len(alg_names)])
    ax5.set_xlabel('综合得分')
    ax5.set_title('算法性能排名')
    ax5.set_xlim(0, 1)
    
    # 添加得分标签
    for bar, score in zip(bars, alg_scores):
        width = bar.get_width()
        ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center')
    
    # 6. 效率vs性能散点图
    ax6 = axes[1, 2]
    
    accuracies = [all_results[alg]['accuracy'] for alg in algorithms]
    times = [all_results[alg]['training_time'] for alg in algorithms]
    
    scatter = ax6.scatter(times, accuracies, s=100, alpha=0.7, c=range(len(algorithms)), cmap='viridis')
    
    for i, alg in enumerate(algorithms):
        ax6.annotate(alg, (times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax6.set_xlabel('训练时间 (秒)')
    ax6.set_ylabel('准确率')
    ax6.set_title('效率 vs 性能')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("\n" + "="*60)
    print("详细性能对比结果")
    print("="*60)
    
    for alg, result in all_results.items():
        print(f"\n{alg}:")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  精确率: {result['precision']:.4f}")
        print(f"  召回率: {result['recall']:.4f}")
        print(f"  F1分数: {result['f1_score']:.4f}")
        print(f"  训练时间: {result['training_time']:.2f}秒")
    
    print("\n" + "="*60)
    print("算法优势分析")
    print("="*60)
    print("量子混合模型优势:")
    print("- 利用量子计算的并行性和纠缠特性")
    print("- 在特定问题上可能具有指数级加速")
    print("- 结合经典和量子计算的优势")
    print("\n经典算法优势:")
    print("- 成熟稳定，易于实现和调试")
    print("- 计算资源需求相对较低")
    print("- 在当前硬件条件下性能可靠")

def main():
    """
    主函数：运行完整的算法对比实验
    """
    print("开始量子经典混合算法 vs 经典算法对比实验")
    print("="*60)
    
    # 加载数据
    print("加载数据集...")
    X_train, X_test, y_train, y_test, scaler, pca = load_deepfake_dataset(
        sample_size=500,
        use_pca=True,
        n_components=6
    )
    
    print(f"数据集信息:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  特征维度: {X_train.shape[1]}")
    
    # 运行经典算法
    classical_results = run_classical_methods(X_train, X_test, y_train, y_test)
    
    # 运行量子混合算法
    hybrid_result, hybrid_model = run_hybrid_quantum_model(X_train, X_test, y_train, y_test)
    
    # 绘制对比图表
    print("\n生成对比图表...")
    plot_comparison(classical_results, hybrid_result)
    
    # 显示量子模型训练历史
    print("\n显示量子混合模型训练历史...")
    hybrid_model.plot_training_history()
    
    print("\n实验完成！")

if __name__ == "__main__":
    main()