"""
用户友好的模型封装
===================

这个文件提供了简化的模型接口，让用户无需了解复杂的技术细节，
就能轻松使用 CausalQwen 进行分类和回归任务。

主要特点：
- 自动处理数据预处理
- 预设最佳参数配置
- 简化的训练和预测接口
- 自动结果可视化
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class SimpleCausalClassifier:
    """
    简化的因果推理分类器
    
    这是一个用户友好的分类器，封装了所有复杂的技术细节。
    您只需要提供数据，剩下的都会自动处理。
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化分类器
        
        Args:
            random_state: 随机种子，确保结果可重复
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.class_names = None
        
        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def fit(self, X, y, validation_split: float = 0.2, epochs: int = 50, verbose: bool = True):
        """
        训练模型
        
        Args:
            X: 特征数据 (可以是 numpy 数组或 pandas DataFrame)
            y: 标签数据 (可以是 numpy 数组或 pandas Series)
            validation_split: 验证集比例，用于监控训练过程
            epochs: 训练轮数
            verbose: 是否显示训练过程
        """
        if verbose:
            print("🚀 开始训练因果推理分类器...")
        
        # 数据预处理
        X = np.array(X)
        y = np.array(y)
        
        # 保存特征和类别名称（如果有）
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        if hasattr(y, 'unique'):
            self.class_names = list(y.unique())
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded, 
            test_size=validation_split, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # 创建因果推理模型
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_encoded))
        
        self.model = self._create_causal_model(input_size, num_classes)
        
        # 训练模型
        train_losses, val_losses, val_accuracies = self._train_model(
            X_train, y_train, X_val, y_val, epochs, verbose
        )
        
        self.is_fitted = True
        
        if verbose:
            final_acc = val_accuracies[-1] if val_accuracies else 0
            print(f"✅ 训练完成！最终验证准确率: {final_acc:.4f}")
            
            # 绘制训练过程
            self._plot_training_history(train_losses, val_losses, val_accuracies)
        
        return self
    
    def predict(self, X, return_probabilities: bool = False, temperature: float = 1.0):
        """
        进行预测
        
        Args:
            X: 要预测的特征数据
            return_probabilities: 是否返回预测概率
            temperature: 推理温度（0=确定性，1=标准，>1=更随机）
        
        Returns:
            预测结果（标签或概率）
        """
        if not self.is_fitted:
            raise ValueError("模型还没有训练！请先调用 fit() 方法。")
        
        # 数据预处理
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # 进行预测
        self.model.eval()
        with torch.no_grad():
            # 模拟因果推理输出
            logits = self.model(X_tensor)
            
            # 应用温度调节
            if temperature != 1.0:
                logits = logits / temperature
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # 转换回原始标签
        pred_labels = self.label_encoder.inverse_transform(predictions.numpy())
        
        if return_probabilities:
            return pred_labels, probabilities.numpy()
        else:
            return pred_labels
    
    def predict_with_explanation(self, X, feature_names: Optional[list] = None):
        """
        预测并提供简单的解释
        
        Args:
            X: 要预测的特征数据
            feature_names: 特征名称列表
        
        Returns:
            预测结果和特征重要性
        """
        predictions, probabilities = self.predict(X, return_probabilities=True)
        
        # 简单的特征重要性分析（基于权重）
        feature_importance = self._get_feature_importance()
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = np.max(probs)
            
            result = {
                'prediction': pred,
                'confidence': confidence,
                'probabilities': dict(zip(self.label_encoder.classes_, probs)),
                'top_features': self._get_top_features(X[i], feature_importance, feature_names)
            }
            results.append(result)
        
        return results
    
    def _create_causal_model(self, input_size: int, num_classes: int):
        """创建简化的因果推理模型"""
        # 这里使用一个简化的神经网络来模拟因果推理
        # 在实际实现中，这里会是真正的 CausalEngine
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        return model
    
    def _train_model(self, X_train, y_train, X_val, y_val, epochs, verbose):
        """训练模型"""
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val).float().mean()
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Val Loss: {val_loss.item():.4f}, "
                      f"Val Acc: {val_acc.item():.4f}")
        
        return train_losses, val_losses, val_accuracies
    
    def _plot_training_history(self, train_losses, val_losses, val_accuracies):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(train_losses, label='训练损失', alpha=0.8)
        ax1.plot(val_losses, label='验证损失', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练过程 - 损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(val_accuracies, label='验证准确率', color='green', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('训练过程 - 准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _get_feature_importance(self):
        """获取特征重要性（简化版本）"""
        if self.model is None:
            return None
        
        # 获取第一层权重作为特征重要性的简单近似
        first_layer = list(self.model.children())[0]
        weights = first_layer.weight.data.abs().mean(dim=0)
        return weights.numpy()
    
    def _get_top_features(self, x, importance, feature_names, top_k=3):
        """获取对当前预测最重要的特征"""
        if importance is None:
            return []
        
        # 计算特征贡献（特征值 × 重要性）
        contributions = np.abs(x * importance)
        top_indices = np.argsort(contributions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"特征_{idx}"
            results.append({
                'feature': name,
                'value': x[idx],
                'importance': importance[idx],
                'contribution': contributions[idx]
            })
        
        return results


class SimpleCausalRegressor:
    """
    简化的因果推理回归器
    
    用于回归任务的用户友好接口。
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化回归器
        
        Args:
            random_state: 随机种子，确保结果可重复
        """
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
        # 设置随机种子
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def fit(self, X, y, validation_split: float = 0.2, epochs: int = 50, verbose: bool = True):
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标数据
            validation_split: 验证集比例
            epochs: 训练轮数
            verbose: 是否显示训练过程
        """
        if verbose:
            print("🚀 开始训练因果推理回归器...")
        
        # 数据预处理
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y).flatten()
        
        # 划分数据集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled,
            test_size=validation_split,
            random_state=self.random_state
        )
        
        # 创建模型
        input_size = X_train.shape[1]
        self.model = self._create_causal_model(input_size)
        
        # 训练
        train_losses, val_losses, val_r2s = self._train_model(
            X_train, y_train, X_val, y_val, epochs, verbose
        )
        
        self.is_fitted = True
        
        if verbose:
            final_r2 = val_r2s[-1] if val_r2s else 0
            print(f"✅ 训练完成！最终验证 R²: {final_r2:.4f}")
            
            # 绘制训练过程
            self._plot_training_history(train_losses, val_losses, val_r2s)
        
        return self
    
    def predict(self, X, return_uncertainty: bool = False, temperature: float = 1.0):
        """
        进行预测
        
        Args:
            X: 要预测的特征数据
            return_uncertainty: 是否返回不确定性估计
            temperature: 推理温度
        
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型还没有训练！请先调用 fit() 方法。")
        
        X = np.array(X)
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor)
            
            # 如果返回不确定性，进行多次采样
            if return_uncertainty:
                samples = []
                for _ in range(100):  # 100次采样
                    sample = self.model(X_tensor)
                    samples.append(sample)
                
                samples = torch.stack(samples)
                mean_pred = samples.mean(dim=0)
                std_pred = samples.std(dim=0)
                
                # 反标准化
                predictions = self.scaler_y.inverse_transform(mean_pred.numpy().reshape(-1, 1)).flatten()
                uncertainties = std_pred.numpy().flatten() * self.scaler_y.scale_
                
                return predictions, uncertainties
            else:
                # 反标准化
                predictions = self.scaler_y.inverse_transform(
                    predictions_scaled.numpy().reshape(-1, 1)
                ).flatten()
                
                return predictions
    
    def _create_causal_model(self, input_size: int):
        """创建简化的因果推理模型"""
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        return model
    
    def _train_model(self, X_train, y_train, X_val, y_val, epochs, verbose):
        """训练模型"""
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        val_r2s = []
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train).flatten()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).flatten()
                val_loss = criterion(val_outputs, y_val)
                
                # 计算 R²
                y_mean = y_val.mean()
                ss_tot = ((y_val - y_mean) ** 2).sum()
                ss_res = ((y_val - val_outputs) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_r2s.append(r2.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Val Loss: {val_loss.item():.4f}, "
                      f"Val R²: {r2.item():.4f}")
        
        return train_losses, val_losses, val_r2s
    
    def _plot_training_history(self, train_losses, val_losses, val_r2s):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(train_losses, label='训练损失', alpha=0.8)
        ax1.plot(val_losses, label='验证损失', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('训练过程 - 损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² 曲线
        ax2.plot(val_r2s, label='验证 R²', color='green', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.set_title('训练过程 - R²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compare_with_sklearn(X, y, task_type='classification', test_size=0.2, random_state=42):
    """
    与 scikit-learn 模型进行对比
    
    Args:
        X: 特征数据
        y: 标签数据
        task_type: 任务类型 ('classification' 或 'regression')
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        对比结果
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    
    print(f"🔄 开始与 scikit-learn 模型对比 ({task_type})")
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    results = {}
    
    if task_type == 'classification':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # CausalQwen
        causal_model = SimpleCausalClassifier(random_state=random_state)
        causal_model.fit(X_train, y_train, verbose=False)
        causal_pred = causal_model.predict(X_test)
        causal_acc = accuracy_score(y_test, causal_pred)
        
        # Random Forest
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        
        results = {
            'CausalQwen': causal_acc,
            'Random Forest': rf_acc,
            'Logistic Regression': lr_acc
        }
        
        print("📊 分类准确率对比:")
        for model, acc in results.items():
            print(f"  {model}: {acc:.4f}")
    
    else:  # regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        
        # CausalQwen
        causal_model = SimpleCausalRegressor(random_state=random_state)
        causal_model.fit(X_train, y_train, verbose=False)
        causal_pred = causal_model.predict(X_test)
        causal_r2 = r2_score(y_test, causal_pred)
        
        # Random Forest
        rf_model = RandomForestRegressor(random_state=random_state)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        
        results = {
            'CausalQwen': causal_r2,
            'Random Forest': rf_r2,
            'Linear Regression': lr_r2
        }
        
        print("📊 回归 R² 对比:")
        for model, r2 in results.items():
            print(f"  {model}: {r2:.4f}")
    
    return results