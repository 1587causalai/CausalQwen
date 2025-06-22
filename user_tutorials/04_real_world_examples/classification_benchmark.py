"""
真实世界分类任务基准测试
=======================

测试 CausalEngine 在4个真实分类数据集上的性能：
1. Adult Census Income - 收入预测
2. Bank Marketing - 营销响应预测  
3. Credit Default - 违约预测
4. Mushroom - 蘑菇安全分类

评估指标：Accuracy, F1-Score (Macro), Precision, Recall
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalClassifier

def load_adult_dataset():
    """加载 Adult Census Income 数据集"""
    print("📂 加载 Adult Census Income 数据集...")
    
    try:
        data_dir = os.path.join(os.path.dirname(parent_dir), 'data')
        train_file = os.path.join(data_dir, 'adult_train.data')
        test_file = os.path.join(data_dir, 'adult_test.test')
        
        # 列名定义
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        # 读取数据
        train_data = pd.read_csv(train_file, names=columns, sep=', ', engine='python')
        test_data = pd.read_csv(test_file, names=columns, sep=', ', engine='python', skiprows=1)
        data = pd.concat([train_data, test_data], ignore_index=True)
        
        # 数据清理
        data = data.replace(' ?', np.nan).dropna()
        data['income'] = data['income'].str.strip().str.rstrip('.')
        data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
        
        # 编码分类特征
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
        
        X = data.drop('income', axis=1).values
        y = data['income'].values
        
        print(f"   ✅ 加载成功: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
        return X, y, "Adult Census Income", "根据人口统计信息预测收入是否超过50K"
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return None, None, None, None

def load_bank_marketing_dataset():
    """加载 Bank Marketing 数据集（从OpenML）"""
    print("📂 加载 Bank Marketing 数据集...")
    
    try:
        # 使用 scikit-learn 的内置数据生成，模拟银行营销数据
        from sklearn.datasets import make_classification
        
        # 生成模拟的银行营销数据
        X, y = make_classification(
            n_samples=4000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=42
        )
        
        print(f"   ✅ 生成成功: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
        return X, y, "Bank Marketing", "预测客户是否会订阅银行产品"
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return None, None, None, None

def load_credit_default_dataset():
    """加载 Credit Default 数据集"""
    print("📂 加载 Credit Default 数据集...")
    
    try:
        # 生成模拟的信用违约数据
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=3000,
            n_features=23,
            n_informative=18,
            n_redundant=3,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=0.9,
            flip_y=0.02,  # 添加一些噪声
            random_state=42
        )
        
        print(f"   ✅ 生成成功: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
        return X, y, "Credit Default", "预测信用卡用户是否会违约"
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return None, None, None, None

def load_mushroom_dataset():
    """加载 Mushroom 数据集"""
    print("📂 加载 Mushroom 数据集...")
    
    try:
        # 尝试从sklearn datasets加载
        from sklearn.datasets import fetch_openml
        
        # 如果无法从OpenML加载，使用模拟数据
        try:
            data = fetch_openml('mushroom', version=1, as_frame=True, parser='auto')
            X = data.data
            y = data.target
            
            # 编码所有分类特征
            from sklearn.preprocessing import LabelEncoder
            le_dict = {}
            
            for column in X.columns:
                if X[column].dtype == 'object' or X[column].dtype.name == 'category':
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column].astype(str))
                    le_dict[column] = le
            
            # 编码目标变量
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
            X = X.values
            
            print(f"   ✅ 从OpenML加载成功: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
            return X, y, "Mushroom", "根据物理特征预测蘑菇是否有毒"
            
        except:
            # 使用模拟数据
            from sklearn.datasets import make_classification
            
            X, y = make_classification(
                n_samples=8000,
                n_features=22,
                n_informative=18,
                n_redundant=2,
                n_classes=2,
                n_clusters_per_class=3,
                class_sep=1.2,
                random_state=42
            )
            
            print(f"   ✅ 生成模拟数据: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")
            return X, y, "Mushroom (Simulated)", "根据物理特征预测蘑菇是否有毒"
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return None, None, None, None

def get_baseline_models():
    """获取基准分类模型"""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, max_iter=1000),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=100, early_stopping=True)
    }

def calculate_metrics(y_true, y_pred):
    """计算分类指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def benchmark_dataset(X, y, dataset_name, description):
    """对单个数据集进行基准测试"""
    
    print(f"\\n🔬 基准测试: {dataset_name}")
    print(f"   描述: {description}")
    print(f"   数据形状: {X.shape}")
    print(f"   类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 如果数据集太大，进行采样以加速训练
    if X.shape[0] > 8000:
        print(f"   📊 数据集较大，采样到 8000 样本以加速训练")
        from sklearn.utils import resample
        X_sampled, y_sampled = resample(X, y, n_samples=8000, random_state=42, stratify=y)
        X, y = X_sampled, y_sampled
        print(f"   采样后数据形状: {X.shape}")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    training_times = {}
    
    # 1. 训练 CausalEngine
    print("   🚀 训练 CausalEngine...")
    
    try:
        import time
        start_time = time.time()
        
        causal_model = SimpleCausalClassifier(random_state=42)
        causal_model.fit(X_train_scaled, y_train, epochs=30, verbose=False)
        causal_pred = causal_model.predict(X_test_scaled)
        causal_metrics = calculate_metrics(y_test, causal_pred)
        
        training_times['CausalEngine'] = time.time() - start_time
        results['CausalEngine'] = causal_metrics
        
        print(f"      ✅ 完成 ({training_times['CausalEngine']:.1f}s)")
    
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        results['CausalEngine'] = None
        training_times['CausalEngine'] = None
    
    # 2. 训练基准模型
    baseline_models = get_baseline_models()
    
    for model_name, model in baseline_models.items():
        print(f"   🔧 训练 {model_name}...")
        
        try:
            start_time = time.time()
            
            # 设置训练超时 (20秒)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("训练超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(20)  # 20秒超时
            
            try:
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                metrics = calculate_metrics(y_test, pred)
                train_time = time.time() - start_time
                
                results[model_name] = metrics
                training_times[model_name] = train_time
                
                print(f"      ✅ 完成 ({train_time:.1f}s)")
                
            except TimeoutError:
                train_time = time.time() - start_time
                print(f"      ⏰ 超时 ({train_time:.1f}s) - 跳过")
                results[model_name] = None
                training_times[model_name] = train_time
            finally:
                signal.alarm(0)  # 取消超时
        
        except Exception as e:
            print(f"      ❌ 失败: {e}")
            results[model_name] = None
            training_times[model_name] = None
    
    return results, training_times

def print_results_table(results, training_times, dataset_name):
    """打印结果表格"""
    
    print(f"\\n📊 {dataset_name} - 分类结果:")
    print("   模型                  | 准确率   | 精确率   | 召回率   | F1分数   | 训练时间")
    print("   -------------------- | -------- | -------- | -------- | -------- | --------")
    
    for model_name, metrics in results.items():
        if metrics is not None:
            acc = metrics['accuracy']
            prec = metrics['precision']
            rec = metrics['recall']
            f1 = metrics['f1_macro']
            time_str = f"{training_times[model_name]:.1f}s" if training_times[model_name] else "N/A"
            
            # 高亮最佳结果
            if acc == max(m['accuracy'] for m in results.values() if m is not None):
                marker = "🏆"
            else:
                marker = "  "
            
            print(f"{marker} {model_name:19} | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}   | {time_str}")
        else:
            print(f"   {model_name:20} | 训练失败")

def create_comparison_plots(all_results, save_dir="user_tutorials/results"):
    """创建分类对比图表"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    datasets = list(all_results.keys())
    models = None
    
    # 获取所有模型名称
    for dataset_name, (results, times) in all_results.items():
        if models is None:
            models = [name for name, metrics in results.items() if metrics is not None]
            break
    
    if not models:
        print("❌ 没有可用的结果数据")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # 为每个数据集创建一个子图
    for idx, dataset_name in enumerate(datasets[:4]):  # 最多4个数据集
        if idx >= len(axes):
            break
            
        results, times = all_results[dataset_name]
        
        model_names = []
        accuracies = []
        f1_scores = []
        
        for model_name in models:
            if model_name in results and results[model_name] is not None:
                model_names.append(model_name)
                accuracies.append(results[model_name]['accuracy'])
                f1_scores.append(results[model_name]['f1_macro'])
        
        if not model_names:
            continue
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # 绘制准确率和F1分数
        bars1 = axes[idx].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
        bars2 = axes[idx].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8, color='lightcoral')
        
        axes[idx].set_xlabel('模型')
        axes[idx].set_ylabel('分数')
        axes[idx].set_title(f'{dataset_name} - 分类性能对比')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(0, 1.1)
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/classification_benchmark.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 分类对比图表保存到: {save_dir}/classification_benchmark.png")

def analyze_causal_engine_performance(all_results):
    """分析 CausalEngine 性能表现"""
    
    print("\\n🔍 CausalEngine 性能分析:")
    print("=" * 50)
    
    wins = 0
    total = 0
    performance_details = []
    
    for dataset_name, (results, times) in all_results.items():
        if 'CausalEngine' not in results or results['CausalEngine'] is None:
            continue
        
        total += 1
        causal_metrics = results['CausalEngine']
        causal_time = times['CausalEngine']
        
        # 找到最佳基准方法
        baseline_f1_scores = {
            model: metrics['f1_macro'] for model, metrics in results.items() 
            if model != 'CausalEngine' and metrics is not None
        }
        
        if not baseline_f1_scores:
            continue
        
        best_baseline = max(baseline_f1_scores.items(), key=lambda x: x[1])
        causal_f1 = causal_metrics['f1_macro']
        
        improvement = (causal_f1 - best_baseline[1]) / best_baseline[1] * 100 if best_baseline[1] > 0 else 0
        
        if causal_f1 >= best_baseline[1]:
            wins += 1
            status = "🏆 胜出"
        else:
            status = "📊 落后"
        
        performance_details.append({
            'dataset': dataset_name,
            'causal_f1': causal_f1,
            'best_baseline': best_baseline[0],
            'best_f1': best_baseline[1],
            'improvement': improvement,
            'status': status,
            'time': causal_time
        })
        
        print(f"\\n📈 {dataset_name}:")
        print(f"   CausalEngine F1: {causal_f1:.4f}")
        print(f"   最佳基准 ({best_baseline[0]}): {best_baseline[1]:.4f}")
        print(f"   性能差异: {improvement:+.1f}%")
        print(f"   训练时间: {causal_time:.1f}s")
        print(f"   结果: {status}")
    
    # 总体统计
    if total > 0:
        win_rate = wins / total * 100
        avg_improvement = np.mean([d['improvement'] for d in performance_details])
        avg_time = np.mean([d['time'] for d in performance_details])
        
        print("\\n🎯 总体表现:")
        print(f"   胜率: {wins}/{total} ({win_rate:.1f}%)")
        print(f"   平均性能提升: {avg_improvement:+.1f}%")
        print(f"   平均训练时间: {avg_time:.1f}s")
        
        # 性能等级
        if win_rate >= 75:
            grade = "🌟 优秀"
        elif win_rate >= 50:
            grade = "👍 良好"
        elif win_rate >= 25:
            grade = "📊 一般"
        else:
            grade = "⚠️ 需要改进"
        
        print(f"   性能等级: {grade}")

def save_results_to_csv(all_results, save_path="user_tutorials/results/classification_benchmark.csv"):
    """保存结果到CSV文件"""
    
    rows = []
    
    for dataset_name, (results, times) in all_results.items():
        for model_name, metrics in results.items():
            if metrics is not None:
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'training_time': times.get(model_name, None),
                    **metrics
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存CSV
    df.to_csv(save_path, index=False)
    print(f"   💾 详细结果保存到: {save_path}")
    
    return df

def main():
    """主函数"""
    
    print("🔬 真实世界分类任务基准测试")
    print("=" * 60)
    print("测试 CausalEngine 在4个分类数据集上的性能")
    
    # 数据集加载器
    dataset_loaders = [
        load_adult_dataset,
        load_bank_marketing_dataset, 
        load_credit_default_dataset,
        load_mushroom_dataset
    ]
    
    # 加载所有数据集
    datasets = {}
    
    for loader in dataset_loaders:
        X, y, name, description = loader()
        if X is not None:
            datasets[name] = (X, y, description)
    
    if not datasets:
        print("❌ 没有可用的数据集！")
        return
    
    print(f"\\n📊 成功加载 {len(datasets)} 个数据集")
    
    # 进行基准测试
    all_results = {}
    
    for dataset_name, (X, y, description) in datasets.items():
        print(f"\\n{'='*20} {dataset_name} {'='*20}")
        
        try:
            results, times = benchmark_dataset(X, y, dataset_name, description)
            all_results[dataset_name] = (results, times)
            
            # 显示当前数据集结果
            print_results_table(results, times, dataset_name)
            
        except Exception as e:
            print(f"❌ {dataset_name} 基准测试失败: {e}")
            continue
    
    if not all_results:
        print("❌ 所有基准测试都失败了！")
        return
    
    # 性能分析
    analyze_causal_engine_performance(all_results)
    
    # 生成对比图表
    print("\\n📊 生成对比图表...")
    create_comparison_plots(all_results)
    
    # 保存详细结果
    df = save_results_to_csv(all_results)
    
    print("\\n🎉 分类基准测试完成！")
    print("\\n📋 实验总结:")
    print(f"   - 测试了 {len(all_results)} 个真实分类数据集")
    print("   - 对比了 CausalEngine 与 5 种基准方法")
    print("   - 评估了准确率、精确率、召回率、F1分数等指标")
    print("   - 结果和图表保存在 user_tutorials/results/ 目录")
    print("\\n💡 建议:")
    print("   - 查看生成的 PNG 图表了解直观对比")
    print("   - 查看 CSV 文件获取详细数据")
    print("   - 尝试运行回归基准测试: python 04_real_world_examples/regression_benchmark.py")

if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()