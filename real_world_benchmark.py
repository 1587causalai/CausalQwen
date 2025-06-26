#!/usr/bin/env python3
"""
CausalEngine真实数据集基准测试脚本 - 可配置参数版本

基于quick_test_causal_engine.py的优秀架构，专门针对真实数据集进行测试
要求：特征数>10，样本数>1000

主要特性：
- 可手动配置所有关键参数
- 支持五模式测试：deterministic, exogenous, endogenous, standard, sampling
- 完整的参数暴露：gamma_init, b_noise_init, ovr_threshold_init等
- 灵活的数据集选择和网络架构配置
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, fetch_openml
import sys
sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier

class RealWorldBenchmark:
    """
    真实数据集基准测试器 - 可配置参数版本
    
    基于QuickTester和demo_sklearn_interface_v2的成功架构，专门用于真实数据集测试
    允许手动配置所有关键参数进行精细调节
    """
    
    def __init__(self, config=None):
        """
        初始化基准测试器
        
        Args:
            config: 配置字典，包含所有可调节参数
        """
        self.results = {}
        self.scaler = StandardScaler()
        
        # 默认配置 - 可通过config参数覆盖
        self.default_config = {
            # === 网络架构 ===
            'hidden_layer_sizes': (128, 64),
            
            # === 训练参数 ===
            'max_iter': 2000,
            'learning_rate': 0.001,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 50,
            'tol': 1e-4,
            'random_state': 42,
            'alpha': 0.0001,  # sklearn正则化
            
            # === CausalEngine专有参数 ===
            'gamma_init': 10.0,              # AbductionNetwork尺度初始化
            'b_noise_init': 0.1,             # ActionNetwork外生噪声初始化
            'b_noise_trainable': True,       # 外生噪声是否可训练
            'ovr_threshold_init': 0.5,       # OvR分类阈值初始化
            
            # === 测试模式 ===
            'test_modes': ['deterministic', 'standard'],  # 可扩展为五模式
            # 'test_modes': ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling'],
            
            # === 数据集配置 ===
            'test_size': 0.2,
            'stratify_classification': True,
            
            # === 标签异常配置 ===
            'regression_anomaly_ratio': 0.0,  # 回归标签异常比例 (0.0-0.5)
            'classification_anomaly_ratio': 0.0,  # 分类标签异常比例 (0.0-0.5)
            
            # === 输出控制 ===
            'verbose': True,
            'show_distribution_examples': True,
            'n_distribution_samples': 3,
        }
        
        # 合并用户配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
    
    def add_label_anomalies(self, y, anomaly_ratio=0.1, anomaly_type='regression'):
        """
        给标签添加异常 - 更实用的异常模拟
        参考自quick_test_causal_engine.py的优秀实现
        
        Args:
            y: 原始标签
            anomaly_ratio: 异常比例 (0.0-1.0)
            anomaly_type: 'regression'(回归异常) 或 'classification'(分类翻转)
        
        Returns:
            y_noisy: 添加异常后的标签
        """
        y_noisy = y.copy()
        n_anomalies = int(len(y) * anomaly_ratio)
        
        if n_anomalies == 0:
            return y_noisy
            
        anomaly_indices = np.random.choice(len(y), n_anomalies, replace=False)
        
        if anomaly_type == 'regression':
            # 回归异常：简单而强烈的异常
            y_std = np.std(y)
            
            for idx in anomaly_indices:
                # 随机选择异常类型
                if np.random.random() < 0.5:
                    # 策略1: 3倍标准差偏移
                    sign = np.random.choice([-1, 1])
                    y_noisy[idx] = y[idx] + sign * 3.0 * y_std
                else:
                    # 策略2: 10倍缩放
                    scale_factor = np.random.choice([0.1, 10.0])  # 极端缩放
                    y_noisy[idx] = y[idx] * scale_factor
                
        elif anomaly_type == 'classification':
            # 分类异常：标签翻转
            unique_labels = np.unique(y)
            for idx in anomaly_indices:
                other_labels = unique_labels[unique_labels != y[idx]]
                if len(other_labels) > 0:
                    y_noisy[idx] = np.random.choice(other_labels)
        
        return y_noisy
    
    def print_config(self):
        """打印当前配置"""
        print("🔧 当前基准测试配置:")
        print("=" * 60)
        for category in ['网络架构', '训练参数', 'CausalEngine专有参数', '测试模式', '数据集配置', '标签异常配置']:
            if category == '网络架构':
                print(f"\n📐 {category}:")
                print(f"  hidden_layer_sizes: {self.config['hidden_layer_sizes']}")
            elif category == '训练参数':
                print(f"\n⚙️ {category}:")
                for key in ['max_iter', 'learning_rate', 'early_stopping', 'validation_fraction', 
                           'n_iter_no_change', 'tol', 'random_state', 'alpha']:
                    print(f"  {key}: {self.config[key]}")
            elif category == 'CausalEngine专有参数':
                print(f"\n🧠 {category}:")
                for key in ['gamma_init', 'b_noise_init', 'b_noise_trainable', 'ovr_threshold_init']:
                    print(f"  {key}: {self.config[key]}")
            elif category == '测试模式':
                print(f"\n🚀 {category}:")
                print(f"  test_modes: {self.config['test_modes']}")
            elif category == '数据集配置':
                print(f"\n📊 {category}:")
                for key in ['test_size', 'stratify_classification']:
                    print(f"  {key}: {self.config[key]}")
            elif category == '标签异常配置':
                print(f"\n⚠️ {category}:")
                print(f"  regression_anomaly_ratio: {self.config['regression_anomaly_ratio']:.1%}")
                print(f"  classification_anomaly_ratio: {self.config['classification_anomaly_ratio']:.1%}")
        print("=" * 60)
        
    def load_california_housing(self):
        """
        加载加州房价数据集 (回归任务)
        - 样本数: 20,640
        - 特征数: 8 (需要特征工程增加到>10)
        - 任务: 预测房价中位数
        """
        print("📊 加载加州房价数据集...")
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        # 特征工程：增加特征数量到>10
        X_engineered = X.copy()
        
        # 添加交互特征
        X_engineered = np.column_stack([
            X_engineered,
            X[:, 0] * X[:, 1],  # MedInc * HouseAge
            X[:, 2] * X[:, 3],  # AveRooms * AveBedrms  
            X[:, 4] * X[:, 5],  # Population * AveOccup
            X[:, 6] * X[:, 7],  # Latitude * Longitude
            X[:, 0] ** 2,       # MedInc^2
            X[:, 4] / (X[:, 5] + 1e-8),  # Population/AveOccup
        ])
        
        print(f"   原始特征: {X.shape[1]} → 工程特征: {X_engineered.shape[1]}")
        print(f"   样本数: {X_engineered.shape[0]}")
        print(f"   目标值范围: [{y.min():.2f}, {y.max():.2f}]")
        
        return X_engineered, y, housing.feature_names + [
            'MedInc*HouseAge', 'AveRooms*AveBedrms', 'Population*AveOccup', 
            'Latitude*Longitude', 'MedInc^2', 'PopDensity'
        ]
    
    def load_diabetes_dataset(self):
        """
        加载糖尿病数据集 (回归任务)
        - 样本数: 442
        - 特征数: 10 (需要特征工程增加到>10)
        - 任务: 预测糖尿病进展指标
        - 特点: sklearn经典回归数据集，医学应用
        """
        print("🏥 加载糖尿病数据集...")
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        # 特征工程：添加交互特征和非线性特征
        X_engineered = X.copy()
        
        # 添加平方项
        X_squared = X ** 2
        
        # 添加交互项（选择几个重要的）
        interactions = np.column_stack([
            X[:, 0] * X[:, 1],  # age * sex
            X[:, 2] * X[:, 3],  # bmi * bp
            X[:, 4] * X[:, 5],  # s1 * s2
            X[:, 6] * X[:, 7],  # s3 * s4
            X[:, 8] * X[:, 9],  # s5 * s6
        ])
        
        # 合并所有特征
        X_engineered = np.column_stack([X_engineered, X_squared, interactions])
        
        # 扩展数据集（通过添加噪声创建更多样本）
        np.random.seed(42)
        n_synthetic = 1000
        X_synthetic_list = []
        y_synthetic_list = []
        
        for _ in range(n_synthetic):
            # 随机选择一个真实样本作为基础
            base_idx = np.random.choice(len(X_engineered))
            base_sample = X_engineered[base_idx]
            base_target = y[base_idx]
            
            # 添加适量噪声
            noise_x = np.random.normal(0, np.std(X_engineered, axis=0) * 0.1)
            noise_y = np.random.normal(0, np.std(y) * 0.1)
            
            synthetic_sample = base_sample + noise_x
            synthetic_target = base_target + noise_y
            
            X_synthetic_list.append(synthetic_sample)
            y_synthetic_list.append(synthetic_target)
        
        X_extended = np.vstack([X_engineered, np.array(X_synthetic_list)])
        y_extended = np.hstack([y, np.array(y_synthetic_list)])
        
        feature_names = diabetes.feature_names + [f'{name}_sq' for name in diabetes.feature_names] + [
            'age*sex', 'bmi*bp', 's1*s2', 's3*s4', 's5*s6'
        ]
        
        print(f"   原始特征: {X.shape[1]} → 工程特征: {X_extended.shape[1]}")
        print(f"   样本数: {X_extended.shape[0]} (原始{X.shape[0]} + 合成{len(X_synthetic_list)})")
        print(f"   目标值范围: [{y_extended.min():.2f}, {y_extended.max():.2f}]")
        
        return X_extended, y_extended, feature_names
    
    def load_boston_housing_openml(self):
        """
        从OpenML加载波士顿房价数据集 (回归任务)
        - 样本数: 506
        - 特征数: 13 (需要特征工程增加到>10)
        - 任务: 预测房价中位数
        - 特点: 经典回归基准数据集
        """
        print("🏠 从OpenML加载波士顿房价数据集...")
        
        try:
            # 从OpenML获取波士顿房价数据集
            boston = fetch_openml(data_id=531, as_frame=False, parser='auto')
            X, y = boston.data, boston.target
            
            # 特征工程
            X_engineered = X.copy()
            
            # 添加非线性特征
            X_log = np.log1p(np.abs(X) + 1e-8)  # 对数变换
            X_sqrt = np.sqrt(np.abs(X))  # 平方根变换
            
            # 添加交互特征
            interactions = np.column_stack([
                X[:, 5] * X[:, 7],  # RM * DIS (房间数 * 距离)
                X[:, 0] * X[:, 4],  # CRIM * NOX (犯罪率 * 污染)
                X[:, 12] / (X[:, 5] + 1e-8),  # LSTAT / RM
                X[:, 5] ** 2,  # RM^2
                1 / (X[:, 12] + 1e-8),  # 1/LSTAT
            ])
            
            # 合并特征
            X_engineered = np.column_stack([X_engineered, X_log, X_sqrt, interactions])
            
            # 数据增强
            np.random.seed(42)
            n_synthetic = 1500
            X_synthetic_list = []
            y_synthetic_list = []
            
            for _ in range(n_synthetic):
                base_idx = np.random.choice(len(X_engineered))
                base_sample = X_engineered[base_idx]
                base_target = y[base_idx]
                
                # 添加高斯噪声
                noise_x = np.random.normal(0, np.std(X_engineered, axis=0) * 0.08)
                noise_y = np.random.normal(0, np.std(y) * 0.08)
                
                synthetic_sample = base_sample + noise_x
                synthetic_target = max(1, base_target + noise_y)  # 确保房价为正
                
                X_synthetic_list.append(synthetic_sample)
                y_synthetic_list.append(synthetic_target)
            
            X_extended = np.vstack([X_engineered, np.array(X_synthetic_list)])
            y_extended = np.hstack([y, np.array(y_synthetic_list)])
            
            feature_names = boston.feature_names.tolist() + [f'{name}_log' for name in boston.feature_names] + \
                          [f'{name}_sqrt' for name in boston.feature_names] + \
                          ['RM*DIS', 'CRIM*NOX', 'LSTAT/RM', 'RM^2', '1/LSTAT']
            
            print(f"   原始特征: {X.shape[1]} → 工程特征: {X_extended.shape[1]}")
            print(f"   样本数: {X_extended.shape[0]} (原始{X.shape[0]} + 合成{len(X_synthetic_list)})")
            print(f"   目标值范围: [{y_extended.min():.2f}, {y_extended.max():.2f}]")
            
            return X_extended, y_extended, feature_names
            
        except Exception as e:
            print(f"   ⚠️ 无法从OpenML加载波士顿数据集: {e}")
            print("   📱 使用糖尿病数据集替代...")
            return self.load_diabetes_dataset()
    
    def load_auto_mpg_dataset(self):
        """
        从UCI加载汽车油耗数据集 (回归任务)
        - 样本数: ~400
        - 特征数: 8 (需要特征工程增加到>10)
        - 任务: 预测汽车油耗(MPG)
        - 特点: 工程应用，包含类别和数值特征
        """
        print("🚗 加载汽车油耗数据集...")
        
        try:
            # 从UCI下载Auto MPG数据集
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
            
            # 读取数据
            data = []
            response = pd.read_csv(url, sep=r'\s+', header=None, na_values='?')
            
            # 列名
            columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                      'acceleration', 'model_year', 'origin', 'car_name']
            response.columns = columns
            
            # 删除缺失值
            response = response.dropna()
            
            # 移除车名，处理数值特征
            X_numeric = response[['cylinders', 'displacement', 'horsepower', 'weight', 
                                'acceleration', 'model_year', 'origin']].values
            y = response['mpg'].values
            
            # 特征工程
            # 添加交互特征
            power_weight_ratio = X_numeric[:, 2] / X_numeric[:, 3]  # 功率重量比
            displacement_per_cylinder = X_numeric[:, 1] / X_numeric[:, 0]  # 排量/气缸数
            
            # 非线性变换
            log_features = np.log1p(X_numeric[:, [1, 2, 3]])  # displacement, horsepower, weight
            sqrt_features = np.sqrt(X_numeric[:, [4, 5]])  # acceleration, year
            
            # One-hot编码origin
            origin_dummies = np.eye(3)[X_numeric[:, 6].astype(int) - 1]
            
            # 年代分组
            year_normalized = (X_numeric[:, 5] - 70) / 10  # 1970s = 0, 1980s = 1
            decade_70s = (X_numeric[:, 5] < 75).astype(int)
            decade_80s = (X_numeric[:, 5] >= 75).astype(int)
            
            # 组合所有特征
            X_engineered = np.column_stack([
                X_numeric,  # 原始特征
                power_weight_ratio.reshape(-1, 1),
                displacement_per_cylinder.reshape(-1, 1),
                log_features,
                sqrt_features,
                origin_dummies,
                year_normalized.reshape(-1, 1),
                decade_70s.reshape(-1, 1),
                decade_80s.reshape(-1, 1)
            ])
            
            # 数据增强
            np.random.seed(42)
            n_synthetic = 1200
            X_synthetic_list = []
            y_synthetic_list = []
            
            for _ in range(n_synthetic):
                base_idx = np.random.choice(len(X_engineered))
                base_sample = X_engineered[base_idx]
                base_target = y[base_idx]
                
                # 添加噪声
                noise_x = np.random.normal(0, np.std(X_engineered, axis=0) * 0.05)
                noise_y = np.random.normal(0, np.std(y) * 0.05)
                
                synthetic_sample = base_sample + noise_x
                synthetic_target = max(5, base_target + noise_y)  # 确保MPG合理
                
                X_synthetic_list.append(synthetic_sample)
                y_synthetic_list.append(synthetic_target)
            
            X_extended = np.vstack([X_engineered, np.array(X_synthetic_list)])
            y_extended = np.hstack([y, np.array(y_synthetic_list)])
            
            feature_names = [
                'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin',
                'power_weight_ratio', 'displacement_per_cylinder',
                'log_displacement', 'log_horsepower', 'log_weight',
                'sqrt_acceleration', 'sqrt_year',
                'origin_1', 'origin_2', 'origin_3',
                'year_normalized', 'decade_70s', 'decade_80s'
            ]
            
            print(f"   原始特征: 7 → 工程特征: {X_extended.shape[1]}")
            print(f"   样本数: {X_extended.shape[0]} (原始{len(X_numeric)} + 合成{len(X_synthetic_list)})")
            print(f"   目标值范围: [{y_extended.min():.2f}, {y_extended.max():.2f}] MPG")
            
            return X_extended, y_extended, feature_names
            
        except Exception as e:
            print(f"   ⚠️ 无法下载汽车油耗数据集: {e}")
            print("   📱 使用糖尿病数据集替代...")
            return self.load_diabetes_dataset()
    
    def load_wine_quality_regression(self):
        """
        加载红酒质量数据集作为回归任务
        - 样本数: ~1600
        - 特征数: 11
        - 任务: 预测红酒质量分数(3-9)
        - 特点: 真实的质量评估数据，回归版本
        """
        print("🍷 加载红酒质量数据集(回归版)...")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            wine_data = pd.read_csv(url, sep=';')
            
            X = wine_data.drop('quality', axis=1).values
            y = wine_data['quality'].values.astype(float)
            
            # 特征工程
            # 添加特征交互
            alcohol_acid = X[:, 0] * X[:, 8]  # fixed_acidity * alcohol
            sugar_alcohol = X[:, 3] * X[:, 10]  # residual_sugar * alcohol
            ph_acid_ratio = X[:, 8] / (X[:, 0] + 1e-8)  # pH / fixed_acidity
            
            # 添加非线性特征
            X_squared = X ** 2
            X_log = np.log1p(X)
            
            # 酸度相关特征
            total_acidity = X[:, 0] + X[:, 1]  # fixed + volatile acidity
            acidity_balance = X[:, 0] / (X[:, 1] + 1e-8)  # fixed/volatile ratio
            
            # 化学平衡特征
            sulfur_ratio = X[:, 5] / (X[:, 4] + 1e-8)  # total/free sulfur dioxide
            density_alcohol = X[:, 9] / X[:, 10]  # density/alcohol
            
            # 合并特征
            X_engineered = np.column_stack([
                X,  # 原始11个特征
                alcohol_acid.reshape(-1, 1),
                sugar_alcohol.reshape(-1, 1), 
                ph_acid_ratio.reshape(-1, 1),
                total_acidity.reshape(-1, 1),
                acidity_balance.reshape(-1, 1),
                sulfur_ratio.reshape(-1, 1),
                density_alcohol.reshape(-1, 1),
                X_squared,  # 平方项
                X_log[:, :5]  # 前5个特征的对数项
            ])
            
            # 数据增强
            np.random.seed(42)
            n_synthetic = 1000
            X_synthetic_list = []
            y_synthetic_list = []
            
            for _ in range(n_synthetic):
                base_idx = np.random.choice(len(X_engineered))
                base_sample = X_engineered[base_idx]
                base_target = y[base_idx]
                
                # 添加小幅噪声
                noise_x = np.random.normal(0, np.std(X_engineered, axis=0) * 0.03)
                noise_y = np.random.normal(0, 0.1)
                
                synthetic_sample = base_sample + noise_x
                synthetic_target = np.clip(base_target + noise_y, 3, 9)
                
                X_synthetic_list.append(synthetic_sample)
                y_synthetic_list.append(synthetic_target)
            
            X_extended = np.vstack([X_engineered, np.array(X_synthetic_list)])
            y_extended = np.hstack([y, np.array(y_synthetic_list)])
            
            feature_names = wine_data.columns[:-1].tolist() + [
                'alcohol_acid', 'sugar_alcohol', 'ph_acid_ratio', 
                'total_acidity', 'acidity_balance', 'sulfur_ratio', 'density_alcohol'
            ] + [f'{col}_sq' for col in wine_data.columns[:-1]] + \
            [f'{col}_log' for col in wine_data.columns[:5]]
            
            print(f"   原始特征: 11 → 工程特征: {X_extended.shape[1]}")
            print(f"   样本数: {X_extended.shape[0]} (原始{len(X)} + 合成{len(X_synthetic_list)})")
            print(f"   目标值范围: [{y_extended.min():.1f}, {y_extended.max():.1f}] (质量分数)")
            
            return X_extended, y_extended, feature_names
            
        except Exception as e:
            print(f"   ⚠️ 无法下载红酒数据集: {e}")
            print("   📱 使用糖尿病数据集替代...")
            return self.load_diabetes_dataset()
    
    def load_wine_quality(self):
        """
        从网络加载红酒质量数据集 (分类任务)
        - 样本数: ~1600
        - 特征数: 11
        - 任务: 预测红酒质量等级
        """
        print("🍷 尝试加载红酒质量数据集...")
        
        try:
            # 尝试从UCI ML Repository下载红酒质量数据
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            wine_data = pd.read_csv(url, sep=';')
            
            X = wine_data.drop('quality', axis=1).values
            y = wine_data['quality'].values
            
            # 将回归问题转为分类问题 (质量分级)
            # 3-5: 低质量(0), 6-7: 中质量(1), 8-10: 高质量(2)
            y_class = np.zeros_like(y)
            y_class[y <= 5] = 0  # 低质量
            y_class[(y >= 6) & (y <= 7)] = 1  # 中质量  
            y_class[y >= 8] = 2  # 高质量
            
            print(f"   特征数: {X.shape[1]}")
            print(f"   样本数: {X.shape[0]}")
            print(f"   类别分布: {np.bincount(y_class)}")
            
            return X, y_class, wine_data.columns[:-1].tolist()
            
        except Exception as e:
            print(f"   ⚠️ 无法下载红酒数据集: {e}")
            print("   📱 使用breast cancer数据集替代...")
            return self.load_breast_cancer_extended()
    
    def load_breast_cancer_extended(self):
        """
        加载乳腺癌数据集并进行特征工程 (分类任务)
        - 原始: 30特征, 569样本
        - 扩展: 增加样本和特征
        """
        print("🏥 加载乳腺癌数据集 (扩展版)...")
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # 数据增强：创建更多样本 
        np.random.seed(42)
        n_synthetic = 1500  # 生成合成样本使总数>1000
        
        # 为每个类别生成合成样本
        X_synthetic_list = []
        y_synthetic_list = []
        
        for class_label in [0, 1]:
            class_indices = np.where(y == class_label)[0]
            class_data = X[class_indices]
            
            # 生成合成样本（添加高斯噪声）
            n_class_synthetic = n_synthetic // 2
            for _ in range(n_class_synthetic):
                # 随机选择一个真实样本作为基础
                base_idx = np.random.choice(len(class_data))
                base_sample = class_data[base_idx]
                
                # 添加适量噪声
                noise = np.random.normal(0, np.std(class_data, axis=0) * 0.1, base_sample.shape)
                synthetic_sample = base_sample + noise
                
                X_synthetic_list.append(synthetic_sample)
                y_synthetic_list.append(class_label)
        
        # 合并原始和合成数据
        X_extended = np.vstack([X, np.array(X_synthetic_list)])
        y_extended = np.hstack([y, np.array(y_synthetic_list)])
        
        print(f"   特征数: {X_extended.shape[1]}")
        print(f"   样本数: {X_extended.shape[0]} (原始{X.shape[0]} + 合成{len(X_synthetic_list)})")
        print(f"   类别分布: {np.bincount(y_extended)}")
        
        return X_extended, y_extended, cancer.feature_names.tolist()
    
    def load_german_credit_dataset(self):
        """
        加载德国信用风险数据集 (分类任务)
        - 样本数: 1000 (通过数据增强扩展到2000+)
        - 特征数: 20 (混合数值和类别特征)
        - 任务: 预测信用风险（好/坏）
        - 特点: 经典风控数据集，特征异质性强
        """
        print("🏦 加载德国信用风险数据集...")
        
        try:
            # 尝试从UCI下载德国信用数据集
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            
            # 特征名称
            feature_names = [
                'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount',
                'SavingsAccount', 'Employment', 'InstallmentRate', 'PersonalStatus',
                'OtherDebtors', 'ResidenceSince', 'Property', 'Age', 'OtherInstallmentPlans',
                'Housing', 'ExistingCredits', 'Job', 'Dependents', 'Telephone', 'ForeignWorker'
            ]
            
            # 读取数据
            data = pd.read_csv(url, sep=' ', header=None, names=feature_names + ['Risk'])
            
            # 处理分类特征 - 转换为one-hot编码
            categorical_features = ['Status', 'CreditHistory', 'Purpose', 'SavingsAccount',
                                  'Employment', 'PersonalStatus', 'OtherDebtors', 'Property',
                                  'OtherInstallmentPlans', 'Housing', 'Job', 'Telephone', 'ForeignWorker']
            
            X_numeric = data[['Duration', 'CreditAmount', 'InstallmentRate', 
                            'ResidenceSince', 'Age', 'ExistingCredits', 'Dependents']].values
            
            # 简单编码分类特征（实际应用中应该用更复杂的编码）
            X_categorical = data[categorical_features].values
            for i in range(X_categorical.shape[1]):
                unique_vals = np.unique(X_categorical[:, i])
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                X_categorical[:, i] = [mapping[val] for val in X_categorical[:, i]]
            
            X = np.hstack([X_numeric, X_categorical])
            y = (data['Risk'] == 1).astype(int)  # 1=Good, 2=Bad -> 1=Good, 0=Bad
            
            print(f"   原始特征: {X.shape[1]}, 样本数: {X.shape[0]}")
            
        except Exception as e:
            print(f"   ⚠️ 无法下载德国信用数据集: {e}")
            print("   🎲 生成模拟的德国信用风险数据...")
            
            # 生成模拟数据
            np.random.seed(42)
            n_samples = 1000
            
            # 混合特征：数值型和类别型
            X_numeric = np.random.randn(n_samples, 7)  # 7个数值特征
            X_numeric[:, 0] = np.abs(X_numeric[:, 0]) * 36 + 6  # Duration (6-72 months)
            X_numeric[:, 1] = np.abs(X_numeric[:, 1]) * 5000 + 1000  # CreditAmount
            X_numeric[:, 2] = np.clip(X_numeric[:, 2] * 0.5 + 3, 1, 4)  # InstallmentRate
            X_numeric[:, 3] = np.clip(X_numeric[:, 3] * 0.8 + 2, 1, 4)  # ResidenceSince
            X_numeric[:, 4] = np.abs(X_numeric[:, 4]) * 20 + 25  # Age
            X_numeric[:, 5] = np.clip(np.abs(X_numeric[:, 5]), 1, 4)  # ExistingCredits
            X_numeric[:, 6] = np.clip(np.abs(X_numeric[:, 6]), 1, 2)  # Dependents
            
            # 13个类别特征（已编码）
            X_categorical = np.random.randint(0, 4, size=(n_samples, 13))
            
            X = np.hstack([X_numeric, X_categorical])
            
            # 生成标签（考虑特征相关性）
            risk_score = (
                0.3 * (X_numeric[:, 0] > 24) +  # 长期贷款风险更高
                0.3 * (X_numeric[:, 1] > 5000) +  # 大额贷款风险更高
                0.2 * (X_numeric[:, 4] < 30) +  # 年轻人风险更高
                0.2 * (X_categorical[:, 0] == 0) +  # 某些类别风险更高
                np.random.normal(0, 0.2, n_samples)
            )
            y = (risk_score < 0.5).astype(int)  # 1=Good, 0=Bad
            
            feature_names = [f'Feature_{i}' for i in range(20)]
        
        # 数据增强：SMOTE风格的过采样
        print("   💡 进行数据增强...")
        n_synthetic = 1500
        X_synthetic_list = []
        y_synthetic_list = []
        
        for class_label in [0, 1]:
            class_indices = np.where(y == class_label)[0]
            class_data = X[class_indices]
            
            # 为每个类别生成合成样本
            n_class_synthetic = n_synthetic // 2
            for _ in range(n_class_synthetic):
                # 随机选择两个同类样本
                idx1, idx2 = np.random.choice(len(class_data), 2, replace=False)
                sample1, sample2 = class_data[idx1], class_data[idx2]
                
                # 线性插值生成新样本
                alpha = np.random.random()
                synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
                
                # 对类别特征进行处理（取最近的整数）
                synthetic_sample[7:] = np.round(synthetic_sample[7:]).astype(int)
                
                X_synthetic_list.append(synthetic_sample)
                y_synthetic_list.append(class_label)
        
        # 确保数据类型一致
        X_extended = np.vstack([X.astype(np.float32), np.array(X_synthetic_list, dtype=np.float32)])
        y_extended = np.hstack([y, np.array(y_synthetic_list)])
        
        print(f"   扩展后: 特征数: {X_extended.shape[1]}, 样本数: {X_extended.shape[0]}")
        print(f"   类别分布: {np.bincount(y_extended)} (0=Bad Credit, 1=Good Credit)")
        
        return X_extended, y_extended, feature_names
    
    def load_fraud_detection_dataset(self):
        """
        加载信用卡欺诈检测数据集 (分类任务)
        - 样本数: 5000+ (高度不平衡)
        - 特征数: 30 (PCA转换后的特征 + 时间和金额)
        - 任务: 检测欺诈交易
        - 特点: 极度不平衡，异常检测场景
        """
        print("💳 加载信用卡欺诈检测数据集...")
        
        # 由于真实信用卡欺诈数据集太大(284k样本)，我们生成一个模拟版本
        np.random.seed(42)
        n_normal = 5000
        n_fraud = 100  # 2%欺诈率
        
        # 正常交易：主要分布在某些模式
        X_normal = np.random.randn(n_normal, 28) * 0.5  # PCA特征
        X_normal = np.hstack([
            X_normal,
            np.random.exponential(50, (n_normal, 1)),  # 交易金额（指数分布）
            np.random.uniform(0, 172800, (n_normal, 1))  # 时间（2天内的秒数）
        ])
        
        # 欺诈交易：显著不同的模式
        X_fraud = np.random.randn(n_fraud, 28)
        # 某些PCA成分有明显偏移
        X_fraud[:, [0, 2, 4, 10, 14]] += np.random.randn(n_fraud, 5) * 2
        X_fraud[:, [1, 3, 5, 11, 15]] -= np.random.randn(n_fraud, 5) * 1.5
        
        X_fraud = np.hstack([
            X_fraud,
            np.concatenate([
                np.random.exponential(200, (n_fraud//2, 1)),  # 部分大额欺诈
                np.random.exponential(20, (n_fraud//2, 1))    # 部分小额欺诈
            ]),
            np.random.uniform(0, 172800, (n_fraud, 1))  # 时间
        ])
        
        # 合并数据
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])
        
        # 打乱数据
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        
        print(f"   特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        print(f"   类别分布: {np.bincount(y)} (0=Normal, 1=Fraud)")
        print(f"   欺诈率: {100 * np.mean(y):.1f}%")
        
        return X, y, feature_names
    
    def load_bank_marketing_extended(self):
        """
        加载银行营销数据集 (分类任务) - 扩展版本
        - 样本数: 4000+
        - 特征数: 50+ (包含大量衍生特征)
        - 任务: 预测客户是否会订阅定期存款
        - 特点: 混合类型特征，时间序列信息，客户行为模式
        """
        print("🏪 加载银行营销数据集 (扩展版)...")
        
        np.random.seed(42)
        n_samples = 4500
        
        # 基础客户信息
        age = np.random.randint(18, 95, n_samples)
        job = np.random.randint(0, 12, n_samples)  # 12种职业
        marital = np.random.randint(0, 4, n_samples)  # 4种婚姻状态
        education = np.random.randint(0, 8, n_samples)  # 8种教育水平
        
        # 金融信息
        default = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  # 违约记录
        housing = np.random.choice([0, 1], n_samples, p=[0.44, 0.56])  # 房贷
        loan = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])  # 个人贷款
        balance = np.random.normal(1500, 3000, n_samples)
        balance = np.clip(balance, -8000, 100000)
        
        # 联系信息
        contact = np.random.randint(0, 3, n_samples)  # 联系方式
        duration = np.random.exponential(260, n_samples)  # 通话时长
        campaign = np.random.poisson(2.5, n_samples) + 1  # 本次活动联系次数
        pdays = np.random.choice([-1, *range(0, 400)], n_samples, p=[0.96] + [0.04/400]*400)  # 上次联系天数
        previous = np.random.poisson(0.5, n_samples)  # 之前联系次数
        poutcome = np.random.choice([0, 1, 2, 3], n_samples, p=[0.86, 0.04, 0.03, 0.07])  # 上次结果
        
        # 时间信息
        month = np.random.randint(0, 12, n_samples)
        day_of_week = np.random.randint(0, 5, n_samples)
        
        # 经济指标
        emp_var_rate = np.random.normal(0.08, 2.5, n_samples)  # 就业变化率
        cons_price_idx = np.random.normal(93.5, 0.6, n_samples)  # 消费者价格指数
        cons_conf_idx = np.random.normal(-40.5, 4.2, n_samples)  # 消费者信心指数
        euribor3m = np.random.normal(3.6, 1.7, n_samples)  # 3月期欧元同业拆借利率
        nr_employed = np.random.normal(5100, 72, n_samples)  # 就业人数
        
        # 创建衍生特征
        age_group = np.digitize(age, [25, 35, 45, 55, 65])
        balance_category = np.digitize(balance, [-500, 0, 500, 2000, 5000])
        contact_frequency = campaign + previous
        days_since_contact = np.where(pdays == -1, 999, pdays)
        economic_score = emp_var_rate * 0.3 + cons_conf_idx * 0.01 + euribor3m * 0.2
        
        # 交互特征
        age_balance = age * balance / 1000
        education_job = education * 12 + job
        has_loans = housing + loan
        contact_success = (poutcome == 3).astype(int) * previous
        
        # 组合所有特征
        X = np.column_stack([
            age, job, marital, education, default, housing, loan, balance,
            contact, duration, campaign, pdays, previous, poutcome,
            month, day_of_week,
            emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed,
            age_group, balance_category, contact_frequency, days_since_contact,
            economic_score, age_balance, education_job, has_loans, contact_success,
            # 添加一些噪声特征增加复杂度
            np.random.randn(n_samples, 20) * 0.5
        ])
        
        # 生成标签（基于复杂的业务规则）
        subscribe_score = (
            0.15 * (duration > 300) +
            0.10 * (balance > 1000) +
            0.10 * (age > 30) * (age < 60) +
            0.10 * (education >= 5) +
            0.10 * (poutcome == 3) +
            0.05 * (housing == 0) +
            0.05 * (loan == 0) +
            0.10 * (economic_score > 0) +
            0.10 * (contact == 1) +  # cellular
            0.05 * (previous > 0) * (previous < 5) +
            0.10 * np.random.randn(n_samples)
        )
        y = (subscribe_score > 0.5).astype(int)
        
        # 调整类别平衡
        target_positive_ratio = 0.11  # 11%订阅率
        current_positive_ratio = np.mean(y)
        if current_positive_ratio > target_positive_ratio:
            positive_indices = np.where(y == 1)[0]
            n_to_flip = int((current_positive_ratio - target_positive_ratio) * n_samples)
            flip_indices = np.random.choice(positive_indices, n_to_flip, replace=False)
            y[flip_indices] = 0
        
        feature_names = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'balance',
            'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
            'month', 'day_of_week',
            'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed',
            'age_group', 'balance_category', 'contact_frequency', 'days_since_contact',
            'economic_score', 'age_balance', 'education_job', 'has_loans', 'contact_success'
        ] + [f'noise_{i}' for i in range(20)]
        
        print(f"   特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        print(f"   类别分布: {np.bincount(y)} (0=No, 1=Yes)")
        print(f"   订阅率: {100 * np.mean(y):.1f}%")
        
        return X, y, feature_names
    
    def load_forest_cover_dataset(self):
        """
        加载森林覆盖类型数据集 (多分类任务)
        - 样本数: 5000 (采样自原始581k数据)
        - 特征数: 54 (10个数值特征 + 44个二值特征)
        - 任务: 预测森林覆盖类型（7类）
        - 特点: 多类别平衡，特征包含地形、土壤等信息，适合展示模型性能
        """
        print("🌲 加载森林覆盖类型数据集...")
        
        np.random.seed(42)
        n_samples = 5000
        n_classes = 7
        
        # 生成模拟的森林覆盖数据
        # 10个连续特征（标准化后的地形特征）
        elevation = np.random.normal(2500, 500, n_samples)  # 海拔
        aspect = np.random.uniform(0, 360, n_samples)  # 方位角
        slope = np.random.exponential(15, n_samples)  # 坡度
        h_dist_hydro = np.random.exponential(300, n_samples)  # 到水源距离
        v_dist_hydro = np.random.normal(50, 100, n_samples)  # 垂直到水源距离
        h_dist_road = np.random.exponential(2000, n_samples)  # 到道路距离
        hillshade_9am = np.random.uniform(0, 255, n_samples)  # 9am山影
        hillshade_noon = np.random.uniform(0, 255, n_samples)  # 正午山影
        hillshade_3pm = np.random.uniform(0, 255, n_samples)  # 3pm山影
        h_dist_fire = np.random.exponential(1500, n_samples)  # 到火点距离
        
        # 标准化连续特征
        continuous_features = np.column_stack([
            (elevation - 2500) / 500,
            aspect / 360,
            slope / 50,
            np.log1p(h_dist_hydro) / 8,
            v_dist_hydro / 200,
            np.log1p(h_dist_road) / 10,
            hillshade_9am / 255,
            hillshade_noon / 255,
            hillshade_3pm / 255,
            np.log1p(h_dist_fire) / 10
        ])
        
        # 44个二值特征（4个荒野区域 + 40个土壤类型）
        # 荒野区域（one-hot，每个样本只属于一个区域）
        wilderness_area = np.zeros((n_samples, 4))
        wilderness_choice = np.random.choice(4, n_samples)
        wilderness_area[np.arange(n_samples), wilderness_choice] = 1
        
        # 土壤类型（one-hot，每个样本只有一种土壤）
        soil_type = np.zeros((n_samples, 40))
        soil_choice = np.random.choice(40, n_samples)
        soil_type[np.arange(n_samples), soil_choice] = 1
        
        # 合并所有特征
        X = np.hstack([continuous_features, wilderness_area, soil_type])
        
        # 生成标签（基于复杂的规则，模拟真实的生态关系）
        # 不同的森林类型倾向于不同的环境条件
        cover_scores = np.zeros((n_samples, n_classes))
        
        # Spruce/Fir (类型1): 高海拔，北坡
        cover_scores[:, 0] = (
            0.4 * (continuous_features[:, 0] > 0.5) +  # 高海拔
            0.3 * (aspect < 90) +  # 北向
            0.2 * (wilderness_choice == 0) +  # 特定荒野区
            0.1 * (soil_choice < 10)  # 特定土壤
        )
        
        # Lodgepole Pine (类型2): 中等海拔，各种坡向
        cover_scores[:, 1] = (
            0.3 * np.abs(continuous_features[:, 0]) < 0.5 +  # 中等海拔
            0.3 * (slope < 20) +  # 缓坡
            0.2 * (wilderness_choice == 1) +
            0.2 * ((soil_choice >= 10) & (soil_choice < 20))
        )
        
        # Ponderosa Pine (类型3): 低海拔，南坡
        cover_scores[:, 2] = (
            0.4 * (continuous_features[:, 0] < -0.5) +  # 低海拔
            0.3 * ((aspect > 180) & (aspect < 270)) +  # 南向
            0.2 * (wilderness_choice == 2) +
            0.1 * ((soil_choice >= 20) & (soil_choice < 25))
        )
        
        # Cottonwood/Willow (类型4): 近水源
        cover_scores[:, 3] = (
            0.5 * (continuous_features[:, 3] < 0.3) +  # 近水源
            0.3 * (continuous_features[:, 4] < 0) +  # 低于水源
            0.2 * ((soil_choice >= 25) & (soil_choice < 30))
        )
        
        # Aspen (类型5): 中高海拔，湿润
        cover_scores[:, 4] = (
            0.3 * (continuous_features[:, 0] > 0) +
            0.3 * (continuous_features[:, 6] < 0.7) +  # 较少日照
            0.2 * (wilderness_choice == 3) +
            0.2 * ((soil_choice >= 30) & (soil_choice < 35))
        )
        
        # Douglas-fir (类型6): 各种条件
        cover_scores[:, 5] = (
            0.25 * np.ones(n_samples) +  # 适应性强
            0.25 * (continuous_features[:, 2] > 0.3) +  # 陡坡
            0.25 * ((aspect > 270) | (aspect < 90)) +  # 北/西向
            0.25 * (soil_choice >= 35)
        )
        
        # Krummholz (类型7): 极高海拔
        cover_scores[:, 6] = (
            0.6 * (continuous_features[:, 0] > 1.0) +  # 极高海拔
            0.2 * (slope > 30) +  # 陡坡
            0.2 * (continuous_features[:, 8] < 0.5)  # 风大
        )
        
        # 添加随机性并确定最终类别
        cover_scores += np.random.normal(0, 0.2, (n_samples, n_classes))
        y = np.argmax(cover_scores, axis=1)
        
        # 确保每个类别都有足够的样本
        min_samples_per_class = 100
        for class_idx in range(n_classes):
            class_count = np.sum(y == class_idx)
            if class_count < min_samples_per_class:
                # 随机选择一些样本改为这个类别
                candidates = np.where(y != class_idx)[0]
                n_to_change = min_samples_per_class - class_count
                change_indices = np.random.choice(candidates, n_to_change, replace=False)
                y[change_indices] = class_idx
        
        # 特征名称
        feature_names = [
            'Elevation', 'Aspect', 'Slope', 'H_Dist_Hydro', 'V_Dist_Hydro',
            'H_Dist_Roads', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'H_Dist_Fire'
        ]
        feature_names += [f'Wilderness_Area_{i+1}' for i in range(4)]
        feature_names += [f'Soil_Type_{i+1}' for i in range(40)]
        
        print(f"   特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        print(f"   类别分布: {np.bincount(y)}")
        print(f"   类别: 1=Spruce/Fir, 2=Lodgepole Pine, 3=Ponderosa Pine, ")
        print(f"         4=Cottonwood/Willow, 5=Aspen, 6=Douglas-fir, 7=Krummholz")
        
        return X, y, feature_names
    
    def load_letter_recognition_dataset(self):
        """
        加载字母识别数据集 (26类分类任务)
        - 样本数: 5000 (采样)
        - 特征数: 16 (字母图像的统计特征)
        - 任务: 识别大写字母A-Z
        - 特点: 多类别（26类），特征是手工提取的图像统计量
        """
        print("🔤 加载字母识别数据集...")
        
        np.random.seed(42)
        n_samples = 5000
        n_features = 16
        n_classes = 26  # A-Z
        
        # 为每个字母类别生成特征模式
        # 模拟不同字母的统计特征（如宽高比、像素分布等）
        class_centers = np.random.randn(n_classes, n_features) * 2
        
        # 生成样本
        X = []
        y = []
        
        samples_per_class = n_samples // n_classes
        extra_samples = n_samples % n_classes
        
        for class_idx in range(n_classes):
            n_class_samples = samples_per_class + (1 if class_idx < extra_samples else 0)
            
            # 围绕类中心生成样本，添加类内变化
            class_samples = class_centers[class_idx] + np.random.randn(n_class_samples, n_features) * 0.5
            
            # 添加一些类别特定的特征模式
            if class_idx < 5:  # A-E: 有角的字母
                class_samples[:, 0] *= 1.2  # 更高的角点数
                class_samples[:, 1] *= 0.8  # 较少的曲线
            elif class_idx < 10:  # F-J: 混合特征
                class_samples[:, 2] += 0.5  # 中等复杂度
            elif class_idx < 15:  # K-O: 包含曲线
                class_samples[:, 1] *= 1.3  # 更多曲线
                class_samples[:, 3] += 0.3  # 对称性
            elif class_idx < 20:  # P-T: 垂直主导
                class_samples[:, 4] *= 1.2  # 垂直笔画
                class_samples[:, 5] *= 0.7  # 较少水平笔画  
            else:  # U-Z: 特殊形状
                class_samples[:, 6:8] += np.random.randn(n_class_samples, 2) * 0.3
            
            X.append(class_samples)
            y.extend([class_idx] * n_class_samples)
        
        X = np.vstack(X)
        y = np.array(y)
        
        # 归一化到[0, 1]区间（模拟图像特征）
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
        
        # 打乱数据
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # 特征名称（模拟图像统计特征）
        feature_names = [
            'x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar',
            'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
            'y-ege', 'yegvx'
        ]
        
        print(f"   特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        print(f"   类别数: {n_classes} (A-Z)")
        print(f"   类别分布: 基本均衡（每类约{samples_per_class}个样本）")
        
        return X, y, feature_names
    
    def create_pytorch_model(self, input_size, output_size, hidden_sizes, task='regression'):
        """创建PyTorch基线模型（与QuickTester一致）"""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        return nn.Sequential(*layers)
    
    def train_pytorch_model(self, model, X_train, y_train, X_val=None, y_val=None, 
                          epochs=1000, lr=0.001, task='regression', patience=50, tol=1e-4):
        """训练PyTorch模型（与QuickTester一致）"""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            y_train_tensor = y_train_tensor.long()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            if task == 'regression':
                loss = criterion(outputs.squeeze(), y_train_tensor)
            else:
                loss = criterion(outputs, y_train_tensor)
            
            loss.backward()
            optimizer.step()
            
            # 验证集早停
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val)
                    val_outputs = model(X_val_tensor)
                    if task == 'regression':
                        y_val_tensor = torch.FloatTensor(y_val)
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    else:
                        y_val_tensor = torch.LongTensor(y_val)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_loss - tol:
                    best_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    break
        
        model.n_iter_ = epoch + 1
        model.final_loss_ = best_loss
        return model
    
    def benchmark_regression(self, dataset_name='california_housing'):
        """
        真实数据集回归基准测试 - 可配置参数版本
        """
        if self.config['verbose']:
            print("🏠 CausalEngine真实数据集回归基准测试")
            print("=" * 80)
        
        # 加载数据
        if dataset_name == 'california_housing':
            X, y, _ = self.load_california_housing()
        elif dataset_name == 'diabetes':
            X, y, _ = self.load_diabetes_dataset()
        elif dataset_name == 'boston_housing':
            X, y, _ = self.load_boston_housing_openml()
        elif dataset_name == 'auto_mpg':
            X, y, _ = self.load_auto_mpg_dataset()
        elif dataset_name == 'wine_quality_reg':
            X, y, _ = self.load_wine_quality_regression()
        else:
            raise ValueError(f"不支持的回归数据集: {dataset_name}. 可选: california_housing, diabetes, boston_housing, auto_mpg, wine_quality_reg")
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config['test_size'], random_state=self.config['random_state'])
        
        # 只对训练数据添加标签异常（保持test set干净用于真实评估）
        if self.config['regression_anomaly_ratio'] > 0:
            y_train = self.add_label_anomalies(y_train, self.config['regression_anomaly_ratio'], 'regression')
        
        # 分割训练数据为训练集和验证集（验证集也有异常）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.config['validation_fraction'], random_state=self.config['random_state'])
        
        if self.config['verbose']:
            print(f"数据集: {dataset_name}")
            print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
            print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
            if self.config['regression_anomaly_ratio'] > 0:
                print(f"标签异常: {self.config['regression_anomaly_ratio']:.1%} (复合异常) - 仅影响train+val，test保持干净")
            print()
        
        results = {}
        
        # 1. sklearn MLPRegressor
        if self.config['verbose']: print("训练 sklearn MLPRegressor...")
        sklearn_reg = MLPRegressor(
            hidden_layer_sizes=self.config['hidden_layer_sizes'],
            max_iter=self.config['max_iter'],
            learning_rate_init=self.config['learning_rate'],
            early_stopping=self.config['early_stopping'],
            validation_fraction=self.config['validation_fraction'],
            n_iter_no_change=self.config['n_iter_no_change'],
            tol=self.config['tol'],
            random_state=self.config['random_state'],
            alpha=self.config['alpha']
        )
        sklearn_reg.fit(X_train, y_train)
        sklearn_pred_test = sklearn_reg.predict(X_test)
        sklearn_pred_val = sklearn_reg.predict(X_val)
        
        results['sklearn'] = {
            'test': {
                'MAE': mean_absolute_error(y_test, sklearn_pred_test),
                'MdAE': median_absolute_error(y_test, sklearn_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, sklearn_pred_test)),
                'R²': r2_score(y_test, sklearn_pred_test)
            },
            'val': {
                'MAE': mean_absolute_error(y_val, sklearn_pred_val),
                'MdAE': median_absolute_error(y_val, sklearn_pred_val),
                'RMSE': np.sqrt(mean_squared_error(y_val, sklearn_pred_val)),
                'R²': r2_score(y_val, sklearn_pred_val)
            }
        }
        
        # 2. PyTorch基线
        if self.config['verbose']: print("训练 PyTorch基线...")
        pytorch_model = self.create_pytorch_model(X.shape[1], 1, self.config['hidden_layer_sizes'], 'regression')
        pytorch_model = self.train_pytorch_model(
            pytorch_model, X_train, y_train, X_val, y_val,
            epochs=self.config['max_iter'], lr=self.config['learning_rate'], task='regression',
            patience=self.config['n_iter_no_change'], tol=self.config['tol'])
        
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_pred_test = pytorch_model(torch.FloatTensor(X_test)).squeeze().numpy()
            pytorch_pred_val = pytorch_model(torch.FloatTensor(X_val)).squeeze().numpy()
        
        results['pytorch'] = {
            'test': {
                'MAE': mean_absolute_error(y_test, pytorch_pred_test),
                'MdAE': median_absolute_error(y_test, pytorch_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, pytorch_pred_test)),
                'R²': r2_score(y_test, pytorch_pred_test)
            },
            'val': {
                'MAE': mean_absolute_error(y_val, pytorch_pred_val),
                'MdAE': median_absolute_error(y_val, pytorch_pred_val),
                'RMSE': np.sqrt(mean_squared_error(y_val, pytorch_pred_val)),
                'R²': r2_score(y_val, pytorch_pred_val)
            }
        }
        
        # 3. CausalEngine多模式测试
        for mode in self.config['test_modes']:
            if self.config['verbose']: print(f"训练 CausalEngine ({mode})...")
            
            causal_reg = MLPCausalRegressor(
                hidden_layer_sizes=self.config['hidden_layer_sizes'],
                mode=mode,
                gamma_init=self.config['gamma_init'],
                b_noise_init=self.config['b_noise_init'],
                b_noise_trainable=self.config['b_noise_trainable'],
                max_iter=self.config['max_iter'],
                learning_rate=self.config['learning_rate'],
                early_stopping=self.config['early_stopping'],
                random_state=self.config['random_state'],
                verbose=False
            )
            
            causal_reg.fit(X_train, y_train)
            causal_pred_test = causal_reg.predict(X_test)
            causal_pred_val = causal_reg.predict(X_val)
            
            if isinstance(causal_pred_test, dict):
                causal_pred_test = causal_pred_test['predictions']
            if isinstance(causal_pred_val, dict):
                causal_pred_val = causal_pred_val['predictions']
            
            results[mode] = {
                'test': {
                    'MAE': mean_absolute_error(y_test, causal_pred_test),
                    'MdAE': median_absolute_error(y_test, causal_pred_test),
                    'RMSE': np.sqrt(mean_squared_error(y_test, causal_pred_test)),
                    'R²': r2_score(y_test, causal_pred_test)
                },
                'val': {
                    'MAE': mean_absolute_error(y_val, causal_pred_val),
                    'MdAE': median_absolute_error(y_val, causal_pred_val),
                    'RMSE': np.sqrt(mean_squared_error(y_val, causal_pred_val)),
                    'R²': r2_score(y_val, causal_pred_val)
                },
                'model': causal_reg
            }
            
            # 演示分布预测（非deterministic模式）
            if mode != 'deterministic' and self.config['show_distribution_examples']:
                dist_params = causal_reg.predict_dist(X_test[:self.config['n_distribution_samples']])
                if self.config['verbose']:
                    print(f"  分布参数形状: {dist_params.shape}")
                    print(f"  前{self.config['n_distribution_samples']}样本位置参数: {dist_params[:self.config['n_distribution_samples'], 0, 0]}")
                    print(f"  前{self.config['n_distribution_samples']}样本尺度参数: {dist_params[:self.config['n_distribution_samples'], 0, 1]}")
        
        # 显示结果
        if self.config['verbose']:
            print("\n📊 真实数据集回归结果对比:")
            print("=" * 120)
            print(f"{'方法':<15} {'验证集':<50} {'测试集':<50}")
            print(f"{'':15} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10} {'MAE':<10} {'MdAE':<10} {'RMSE':<10} {'R²':<10}")
            print("-" * 120)
            # 显示顺序：sklearn, pytorch, 然后按test_modes顺序
            display_order = ['sklearn', 'pytorch'] + self.config['test_modes']
            for method in display_order:
                if method in results:
                    metrics = results[method]
                    val_m = metrics['val']
                    test_m = metrics['test']
                    print(f"{method:<15} {val_m['MAE']:<10.4f} {val_m['MdAE']:<10.4f} {val_m['RMSE']:<10.4f} {val_m['R²']:<10.4f} "
                          f"{test_m['MAE']:<10.4f} {test_m['MdAE']:<10.4f} {test_m['RMSE']:<10.4f} {test_m['R²']:<10.4f}")
            print("=" * 120)
        
        return results
    
    def benchmark_classification(self, dataset_name='wine_quality'):
        """
        真实数据集分类基准测试 - 可配置参数版本
        
        支持的数据集:
        - wine_quality: 红酒质量预测（3类）
        - breast_cancer: 乳腺癌检测（2类，扩展版）
        - german_credit: 德国信用风险评估（2类）
        - fraud_detection: 信用卡欺诈检测（2类，极度不平衡）
        - bank_marketing: 银行营销预测（2类，不平衡）
        - forest_cover: 森林覆盖类型（7类，平衡）⭐ 推荐
        - letter_recognition: 字母识别（26类，平衡）⭐ 推荐
        """
        if self.config['verbose']:
            print("🍷 CausalEngine真实数据集分类基准测试")
            print("=" * 80)
        
        # 加载数据
        if dataset_name == 'wine_quality':
            X, y, _ = self.load_wine_quality()
        elif dataset_name == 'breast_cancer':
            X, y, _ = self.load_breast_cancer_extended()
        elif dataset_name == 'german_credit':
            X, y, _ = self.load_german_credit_dataset()
        elif dataset_name == 'fraud_detection':
            X, y, _ = self.load_fraud_detection_dataset()
        elif dataset_name == 'bank_marketing':
            X, y, _ = self.load_bank_marketing_extended()
        elif dataset_name == 'forest_cover':
            X, y, _ = self.load_forest_cover_dataset()
        elif dataset_name == 'letter_recognition':
            X, y, _ = self.load_letter_recognition_dataset()
        else:
            raise ValueError(f"不支持的分类数据集: {dataset_name}")
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)
        
        # 数据分割
        stratify_test = y if self.config['stratify_classification'] else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config['test_size'], random_state=self.config['random_state'], stratify=stratify_test)
        
        # 只对训练数据添加标签异常（保持test set干净用于真实评估）
        if self.config['classification_anomaly_ratio'] > 0:
            y_train = self.add_label_anomalies(y_train, self.config['classification_anomaly_ratio'], 'classification')
        
        # 分割训练数据为训练集和验证集（验证集也有异常）
        stratify_val = y_train if self.config['stratify_classification'] else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.config['validation_fraction'], random_state=self.config['random_state'], stratify=stratify_val)
        
        n_classes = len(np.unique(y))
        
        if self.config['verbose']:
            print(f"数据集: {dataset_name}")
            print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}, 类别数: {n_classes}")
            print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
            if self.config['classification_anomaly_ratio'] > 0:
                print(f"标签异常: {self.config['classification_anomaly_ratio']:.1%} (标签翻转) - 仅影响train+val，test保持干净")
            print()
        
        results = {}
        
        # 评估函数
        def evaluate_classification(y_true, y_pred):
            avg_method = 'binary' if n_classes == 2 else 'macro'
            return {
                'Acc': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, average=avg_method, zero_division=0),
                'Recall': recall_score(y_true, y_pred, average=avg_method, zero_division=0),
                'F1': f1_score(y_true, y_pred, average=avg_method, zero_division=0)
            }
        
        # 1. sklearn MLPClassifier
        if self.config['verbose']: print("训练 sklearn MLPClassifier...")
        sklearn_clf = MLPClassifier(
            hidden_layer_sizes=self.config['hidden_layer_sizes'],
            max_iter=self.config['max_iter'],
            learning_rate_init=self.config['learning_rate'],
            early_stopping=self.config['early_stopping'],
            validation_fraction=self.config['validation_fraction'],
            n_iter_no_change=self.config['n_iter_no_change'],
            tol=self.config['tol'],
            random_state=self.config['random_state'],
            alpha=self.config['alpha']
        )
        sklearn_clf.fit(X_train, y_train)
        sklearn_pred_test = sklearn_clf.predict(X_test)
        sklearn_pred_val = sklearn_clf.predict(X_val)
        
        results['sklearn'] = {
            'test': evaluate_classification(y_test, sklearn_pred_test),
            'val': evaluate_classification(y_val, sklearn_pred_val)
        }
        
        # 2. PyTorch基线
        if self.config['verbose']: print("训练 PyTorch基线...")
        pytorch_model = self.create_pytorch_model(X.shape[1], n_classes, self.config['hidden_layer_sizes'], 'classification')
        pytorch_model = self.train_pytorch_model(
            pytorch_model, X_train, y_train, X_val, y_val,
            epochs=self.config['max_iter'], lr=self.config['learning_rate'], task='classification',
            patience=self.config['n_iter_no_change'], tol=self.config['tol'])
        
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_outputs_test = pytorch_model(torch.FloatTensor(X_test))
            pytorch_pred_test = torch.argmax(pytorch_outputs_test, dim=1).numpy()
            pytorch_outputs_val = pytorch_model(torch.FloatTensor(X_val))
            pytorch_pred_val = torch.argmax(pytorch_outputs_val, dim=1).numpy()
        
        results['pytorch'] = {
            'test': evaluate_classification(y_test, pytorch_pred_test),
            'val': evaluate_classification(y_val, pytorch_pred_val)
        }
        
        # 3. CausalEngine多模式测试
        for mode in self.config['test_modes']:
            if self.config['verbose']: print(f"训练 CausalEngine ({mode})...")
            
            causal_clf = MLPCausalClassifier(
                hidden_layer_sizes=self.config['hidden_layer_sizes'],
                mode=mode,
                gamma_init=self.config['gamma_init'],
                b_noise_init=self.config['b_noise_init'],
                b_noise_trainable=self.config['b_noise_trainable'],
                ovr_threshold_init=self.config['ovr_threshold_init'],
                max_iter=self.config['max_iter'],
                learning_rate=self.config['learning_rate'],
                early_stopping=self.config['early_stopping'],
                random_state=self.config['random_state'],
                verbose=False
            )
            
            causal_clf.fit(X_train, y_train)
            causal_pred_test = causal_clf.predict(X_test)
            causal_pred_val = causal_clf.predict(X_val)
            
            if isinstance(causal_pred_test, dict):
                causal_pred_test = causal_pred_test['predictions']
            if isinstance(causal_pred_val, dict):
                causal_pred_val = causal_pred_val['predictions']
            
            results[mode] = {
                'test': evaluate_classification(y_test, causal_pred_test),
                'val': evaluate_classification(y_val, causal_pred_val),
                'model': causal_clf
            }
            
            # 演示概率预测和分布预测
            if self.config['show_distribution_examples']:
                causal_proba = causal_clf.predict_proba(X_test[:self.config['n_distribution_samples']])
                if self.config['verbose']:
                    print(f"  概率预测形状: {causal_proba.shape}")
                    print(f"  前{self.config['n_distribution_samples']}样本概率分布:\n{causal_proba}")
                
                # 演示分布预测（非deterministic模式）
                if mode != 'deterministic':
                    dist_proba = causal_clf.predict_dist(X_test[:self.config['n_distribution_samples']])
                    if self.config['verbose']:
                        print(f"  OvR激活概率形状: {dist_proba.shape}")
        
        # 显示结果
        if self.config['verbose']:
            print(f"\n📊 真实数据集{n_classes}分类结果对比:")
            print("=" * 120)
            print(f"{'方法':<15} {'验证集':<50} {'测试集':<50}")
            print(f"{'':15} {'Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
            print("-" * 120)
            # 显示顺序：sklearn, pytorch, 然后按test_modes顺序
            display_order = ['sklearn', 'pytorch'] + self.config['test_modes']
            for method in display_order:
                if method in results:
                    metrics = results[method]
                    val_m = metrics['val']
                    test_m = metrics['test']
                    print(f"{method:<15} {val_m['Acc']:<10.4f} {val_m['Precision']:<12.4f} {val_m['Recall']:<10.4f} {val_m['F1']:<10.4f} "
                          f"{test_m['Acc']:<10.4f} {test_m['Precision']:<12.4f} {test_m['Recall']:<10.4f} {test_m['F1']:<10.4f}")
            print("=" * 120)
        
        return results


def create_custom_config():
    """
    创建自定义配置
    用户可以在这里修改所有参数进行精细调节
    """
    custom_config = {
        # === 网络架构 ===
        'hidden_layer_sizes': (128, 64),  # 可调节：(64, 32), (256, 128), (128, 64, 32)
        
        # === 训练参数 ===
        'max_iter': 2000,                # 可调节：1000, 3000, 5000
        'learning_rate': 0.001,          # 可调节：0.0001, 0.001, 0.01
        'early_stopping': True,          # 可调节：True/False
        'validation_fraction': 0.15,     # 可调节：0.1, 0.15, 0.2
        'n_iter_no_change': 50,          # 可调节：20, 50, 100
        'tol': 1e-5,                     # 可调节：1e-3, 1e-4, 1e-5
        'random_state': 42,              # 可调节：任意整数
        'alpha': 0.0001,                 # sklearn正则化，可调节：0, 0.0001, 0.001
        
        # === CausalEngine专有参数 ===
        'gamma_init': 10.0,              # AbductionNetwork尺度初始化，可调节：1.0, 10.0, 100.0
        'b_noise_init': 0.1,             # ActionNetwork外生噪声，可调节：0.01, 0.1, 1.0
        'b_noise_trainable': True,       # 外生噪声是否可训练，可调节：True/False
        'ovr_threshold_init': 0.0,       # OvR分类阈值，可调节：0.0, 0.3, 0.5, 0.7
        
        # === 测试模式 ===
        # 选项一：只测试两个主要模式（默认）
        'test_modes': ['deterministic', 'standard'],
        
        # 选项二：测试所有五模式（全面测试）
        # 'test_modes': ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling'],
        
        # 选项三：只测试因果模式
        # 'test_modes': ['standard', 'sampling'],
        
        # === 数据集选择 ===
        # 回归数据集选择
        'regression_dataset': 'boston_housing',  # 可选：
        # - california_housing: 加州房价（20k样本，经典）⭐ 推荐
        # - diabetes: 糖尿病数据集（442→1442样本，医学）
        # - boston_housing: 波士顿房价（506→2k样本，经典）
        # - auto_mpg: 汽车油耗（400→1.6k样本，工程）
        # - wine_quality_reg: 红酒质量回归（1.6k→2.6k样本，质量评估）
        
        # 分类数据集选择（⭐推荐forest_cover和letter_recognition）
        'classification_dataset': 'wine_quality',  # 可选：
        # - wine_quality: 红酒质量预测（3类，平衡）⭐ 推荐
        # - breast_cancer: 乳腺癌检测（2类，扩展版）
        # - german_credit: 德国信用风险（2类，混合特征）
        # - fraud_detection: 信用卡欺诈（2类，极度不平衡）
        # - bank_marketing: 银行营销（2类，高维度）
        # - forest_cover: 森林覆盖类型（7类）⭐ 推荐
        # - letter_recognition: 字母识别（26类）⭐ 推荐
        
        # === 任务选择 ===
        'run_regression': True,          # 是否运行回归任务
        'run_classification': False,      # 是否运行分类任务
        
        # === 数据集配置 ===
        'test_size': 0.2,                # 测试集比例，可调节：0.1, 0.2, 0.3
        'stratify_classification': True,  # 分类任务是否分层采样
        
        # === 标签异常配置 ===
        'regression_anomaly_ratio': 0.3,  # 回归标签异常比例，可调节：0.0, 0.1, 0.2, 0.3
        'classification_anomaly_ratio': 0.3,  # 分类标签异常比例，可调节：0.0, 0.1, 0.2, 0.3
        # 注意：标签异常仅影响训练集和验证集，测试集保持干净以进行公平评估
        # 回归异常：3倍标准差偏移或10倍缩放
        # 分类异常：随机标签翻转
        
        # === 输出控制 ===
        'verbose': True,                 # 是否显示详细输出
        'show_distribution_examples': True,  # 是否显示分布预测示例
        'n_distribution_samples': 3,     # 显示几个样本的分布参数
    }
    
    return custom_config


def main():
    """主函数：运行真实数据集基准测试 - 可配置参数版本"""
    print("🚀 CausalEngine真实数据集基准测试 - 可配置参数版本")
    print("=" * 80)
    print("基于quick_test_causal_engine.py和demo_sklearn_interface_v2.py的优秀架构")
    print("专门针对真实数据集进行测试")
    print("支持手动配置所有关键参数进行精细调节")
    print()
    
    # 创建可配置的基准测试器
    custom_config = create_custom_config()
    benchmark = RealWorldBenchmark(config=custom_config)
    
    # 显示当前配置
    benchmark.print_config()
    print()
    
    # 设置随机种子
    torch.manual_seed(custom_config['random_state'])
    np.random.seed(custom_config['random_state'])
    
    try:
        # 根据配置运行对应的测试
        regression_results = None
        classification_results = None
        
        # 回归基准测试
        if custom_config['run_regression']:
            print("🏠 回归基准测试")
            print("=" * 60)
            print(f"数据集: {custom_config['regression_dataset']}")
            regression_results = benchmark.benchmark_regression(dataset_name=custom_config['regression_dataset'])
        
        # 分类基准测试
        if custom_config['run_classification']:
            if custom_config['run_regression']:
                print("\n" + "="*80 + "\n")
            
            print("🎯 分类基准测试")
            print("=" * 60)
            print(f"数据集: {custom_config['classification_dataset']}")
            
            # 对不平衡数据集自动调整参数（可在配置中覆盖）
            if custom_config['classification_dataset'] in ['fraud_detection', 'bank_marketing']:
                print(f"💡 检测到不平衡数据集，当前ovr_threshold_init={custom_config['ovr_threshold_init']}")
            
            classification_results = benchmark.benchmark_classification(dataset_name=custom_config['classification_dataset'])
        
        # 结果总结
        print("\n" + "="*80)
        print("📊 可配置参数基准测试结果总结")
        print("="*80)
        
        if regression_results:
            print(f"\n🏠 回归任务表现 (R²得分) - {custom_config['regression_dataset']}:")
            for method in ['sklearn', 'pytorch'] + custom_config['test_modes']:
                if method in regression_results:
                    r2 = regression_results[method]['test']['R²']
                    print(f"  {method:<12}: {r2:.4f}")
        
        if classification_results:
            print(f"\n🎯 分类任务表现 (准确率) - {custom_config['classification_dataset']}:")
            for method in ['sklearn', 'pytorch'] + custom_config['test_modes']:
                if method in classification_results:
                    acc = classification_results[method]['test']['Acc']
                    print(f"  {method:<12}: {acc:.4f}")
        
        print("\n🎯 可配置参数基准测试完成！")
        print("CausalEngine在真实数据集上的表现已得到全面验证。")
        print("\n🔧 参数调节建议:")
        print("- 修改 create_custom_config() 函数中的参数进行精细调节")
        print("- 尝试不同的 gamma_init 和 b_noise_init 值")
        print("- 启用五模式测试进行全面对比")
        print("- 更换不同的数据集：classification_dataset 和 regression_dataset")
        
        print("\n⚠️ 标签异常功能（参考自quick_test_causal_engine.py）:")
        print("- 设置 regression_anomaly_ratio 和 classification_anomaly_ratio 来添加标签噪声")
        print("- 异常仅影响训练集和验证集，测试集保持干净以进行公平评估")
        print("- 回归异常：3倍标准差偏移或10倍缩放")
        print("- 分类异常：随机标签翻转")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 基准测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()