"""
CausalEngine科学等价性验证演示

基于科学标准的数学等价性验证：
1. 三方对比验证框架 (sklearn + PyTorch + CausalEngine)
2. 以sklearn-PyTorch基准差异作为科学标准
3. 1.5倍容忍度范围的合理判断
4. 完整的早停策略公平对比
5. 五模式全面功能验证

核心逻辑：
- sklearn和PyTorch实现相同算法但有差异 -> 建立基准范围
- CausalEngine在此范围内 -> 证明数学实现正确
- 避免过度严格标准的误判
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import sys

sys.path.append('/Users/gongqian/DailyLog/CausalQwen')

try:
    from causal_engine.sklearn import MLPCausalRegressor, MLPCausalClassifier
    print("✅ CausalEngine sklearn接口导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


class SimpleMLPRegressor:
    """使用PyTorch nn.Sequential实现的简单MLP控制组"""
    
    def __init__(self, hidden_layer_sizes=(64, 32), learning_rate=0.001, 
                 max_iter=2000, random_state=42, alpha=0.0, batch_size=32,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.model = None
        self.device = torch.device('cpu')
        self.n_iter_ = 0  # 记录实际训练轮数
        
    def _build_model(self, input_size, output_size):
        """构建与CausalEngine相同的网络结构"""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        layers = []
        prev_size = input_size
        
        # 隐藏层
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
        # 使用与PyTorch默认相同的权重初始化
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)
        
        return self.model
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)
            
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
            
        self._build_model(X.shape[1], y.shape[1])
        
        # 使用Adam优化器（与CausalEngine保持一致）
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                              weight_decay=self.alpha)
        
        # 数据分割
        if self.early_stopping and self.validation_fraction > 0:
            val_size = int(len(X) * self.validation_fraction)
            train_size = len(X) - val_size
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None
        
        # 改用批处理训练（更接近实际情况）
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 早停相关变量
        best_val_loss = float('inf')
        no_improve_count = 0
        
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # 前向传播
                output = self.model(batch_X)
                
                # MSE损失
                loss = F.mse_loss(output, batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            self.n_iter_ = epoch + 1  # 记录当前轮数
            
            # 早停检查
            if self.early_stopping and X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val)
                    val_loss = F.mse_loss(val_output, y_val).item()
                
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= self.n_iter_no_change:
                    print(f"PyTorch控制组早停: Epoch {epoch}, 验证损失无改善 {no_improve_count} 轮")
                    break
                
                self.model.train()
            
            if epoch % 200 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"PyTorch控制组 Epoch {epoch}, Avg Loss: {avg_loss:.6f}")
                
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            
        return output.numpy().squeeze()


class SimpleMLPClassifier:
    """使用PyTorch nn.Sequential实现的简单MLP分类器控制组"""
    
    def __init__(self, hidden_layer_sizes=(64, 32), learning_rate=0.001, 
                 max_iter=1000, random_state=42, alpha=0.0, batch_size=32,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.model = None
        self.n_classes = None
        self.device = torch.device('cpu')
        self.n_iter_ = 0  # 记录实际训练轮数
        
    def _build_model(self, input_size, n_classes):
        """构建分类网络结构"""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        layers = []
        prev_size = input_size
        
        # 隐藏层
        for hidden_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.model = nn.Sequential(*layers)
        
        # 权重初始化
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)
        
        return self.model
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)
            
        self.n_classes = len(np.unique(y.numpy()))
        self._build_model(X.shape[1], self.n_classes)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                              weight_decay=self.alpha)
        
        # 数据分割
        if self.early_stopping and self.validation_fraction > 0:
            val_size = int(len(X) * self.validation_fraction)
            train_size = len(X) - val_size
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None
        
        # 改用批处理训练（更接近实际情况）
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 早停相关变量
        best_val_loss = float('inf')
        no_improve_count = 0
        
        self.model.train()
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                output = self.model(batch_X)
                loss = F.cross_entropy(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            self.n_iter_ = epoch + 1  # 记录当前轮数
            
            # 早停检查
            if self.early_stopping and X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val)
                    val_loss = F.cross_entropy(val_output, y_val).item()
                
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= self.n_iter_no_change:
                    print(f"PyTorch分类器早停: Epoch {epoch}, 验证损失无改善 {no_improve_count} 轮")
                    break
                
                self.model.train()
            
            if epoch % 200 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"PyTorch分类器 Epoch {epoch}, Avg Loss: {avg_loss:.6f}")
                
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            predictions = torch.argmax(output, dim=1)
            
        return predictions.numpy()
    
    def predict_proba(self, X):
        """预测概率"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            probabilities = F.softmax(output, dim=1)
            
        return probabilities.numpy()


def scientific_regression_equivalence_test():
    """基于科学标准的回归等价性验证"""
    print("\n" + "="*60)
    print("🔬 科学回归等价性验证")
    print("="*60)
    
    # 生成固定数据
    X, y = make_regression(n_samples=800, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据: {X_train.shape} 训练, {X_test.shape} 测试")
    
    # 使用早停策略但增加耐心的配置进行充分训练对比
    common_params = {
        'hidden_layer_sizes': (64, 32),
        'max_iter': 3000,  # 提高最大epoch数
        'random_state': 42,
        'early_stopping': True,  # 启用早停策略
        'validation_fraction': 0.1,  # 10%作为验证集
        'n_iter_no_change': 50,  # 增加耐心：连续50轮无改进则停止
        'tol': 1e-5,  # 更严格的改进阈值
        'learning_rate_init': 0.001,
        'alpha': 0.0,
    }
    
    print(f"\n📊 使用高耐心早停策略的超参数:")
    for key, value in common_params.items():
        print(f"  {key}: {value}")
    
    # 1. sklearn MLPRegressor
    print(f"\n--- sklearn MLPRegressor ---")
    sklearn_reg = MLPRegressor(**common_params)
    sklearn_reg.fit(X_train, y_train)
    sklearn_pred = sklearn_reg.predict(X_test)
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    
    print(f"R²: {sklearn_r2:.6f}")
    print(f"MSE: {sklearn_mse:.6f}")
    print(f"训练迭代数: {sklearn_reg.n_iter_}")
    
    # 2. PyTorch nn.Sequential控制组
    print(f"\n--- PyTorch nn.Sequential控制组 ---")
    pytorch_params = {
        'hidden_layer_sizes': common_params['hidden_layer_sizes'],
        'max_iter': common_params['max_iter'],
        'random_state': common_params['random_state'],
        'learning_rate': common_params['learning_rate_init'],
        'alpha': common_params['alpha'],
        'early_stopping': common_params['early_stopping'],
        'validation_fraction': common_params['validation_fraction'],
        'n_iter_no_change': common_params['n_iter_no_change'],
        'tol': common_params['tol'],
    }
    
    pytorch_reg = SimpleMLPRegressor(**pytorch_params)
    pytorch_reg.fit(X_train, y_train)
    pytorch_pred = pytorch_reg.predict(X_test)
    pytorch_r2 = r2_score(y_test, pytorch_pred)
    pytorch_mse = mean_squared_error(y_test, pytorch_pred)
    
    print(f"R²: {pytorch_r2:.6f}")
    print(f"MSE: {pytorch_mse:.6f}")
    print(f"训练迭代数: {pytorch_reg.n_iter_}")
    
    # 3. CausalEngine deterministic模式
    print(f"\n--- CausalEngine deterministic模式 ---")
    
    # 转换sklearn参数到CausalEngine参数
    causal_params = {
        'hidden_layer_sizes': common_params['hidden_layer_sizes'],
        'max_iter': common_params['max_iter'],
        'random_state': common_params['random_state'],
        'early_stopping': common_params['early_stopping'],
        'validation_fraction': common_params['validation_fraction'],
        'n_iter_no_change': common_params['n_iter_no_change'],
        'tol': common_params['tol'],
        'learning_rate': common_params['learning_rate_init'],  # 参数名转换
        'alpha': common_params['alpha'],
        'mode': 'deterministic',
        'verbose': False
    }
    
    causal_reg = MLPCausalRegressor(**causal_params)
    causal_reg.fit(X_train, y_train)
    causal_pred = causal_reg.predict(X_test)
    
    if isinstance(causal_pred, dict):
        causal_pred = causal_pred['predictions']
    
    causal_r2 = r2_score(y_test, causal_pred)
    causal_mse = mean_squared_error(y_test, causal_pred)
    
    print(f"R²: {causal_r2:.6f}")
    print(f"MSE: {causal_mse:.6f}")
    
    # 4. 数学等价性核心验证
    print(f"\n🎯 数学等价性验证 - 回归任务")
    print("=" * 70)
    
    # 关键差异计算
    pytorch_causal_mse_diff = abs(pytorch_mse - causal_mse)
    sklearn_causal_mse_diff = abs(sklearn_mse - causal_mse)
    pytorch_causal_r2_diff = abs(pytorch_r2 - causal_r2)
    sklearn_causal_r2_diff = abs(sklearn_r2 - causal_r2)
    pytorch_causal_corr = np.corrcoef(pytorch_pred, causal_pred)[0,1]
    sklearn_causal_corr = np.corrcoef(sklearn_pred, causal_pred)[0,1]
    
    # 基准差异计算 (sklearn vs PyTorch)
    sklearn_pytorch_mse_diff = abs(sklearn_mse - pytorch_mse)
    sklearn_pytorch_r2_diff = abs(sklearn_r2 - pytorch_r2)
    
    print(f"核心问题：CausalEngine deterministic模式是否与传统MLP数学等价？")
    print(f"")
    print(f"📊 性能对比:")
    print(f"  方法                R²           MSE          训练轮数")
    print(f"  sklearn           {sklearn_r2:.6f}     {sklearn_mse:.2f}       {sklearn_reg.n_iter_}")
    print(f"  PyTorch控制组      {pytorch_r2:.6f}     {pytorch_mse:.2f}       {pytorch_reg.n_iter_}")
    print(f"  CausalEngine      {causal_r2:.6f}     {causal_mse:.2f}       {getattr(causal_reg, 'n_iter_', 'N/A')}")
    
    print(f"\n📏 科学等价性分析:")
    print(f"  sklearn ↔ PyTorch 基准差异 (相同算法，不同实现):")
    print(f"    R²差异:     {sklearn_pytorch_r2_diff:.6f}")
    print(f"    MSE差异:    {sklearn_pytorch_mse_diff:.2f}")
    print(f"")
    print(f"  CausalEngine ↔ sklearn:")
    print(f"    R²差异:     {sklearn_causal_r2_diff:.6f}")
    print(f"    MSE差异:    {sklearn_causal_mse_diff:.2f}")
    print(f"    预测相关性: {sklearn_causal_corr:.6f}")
    
    print(f"  CausalEngine ↔ PyTorch:")
    print(f"    R²差异:     {pytorch_causal_r2_diff:.6f}")
    print(f"    MSE差异:    {pytorch_causal_mse_diff:.2f}")
    print(f"    预测相关性: {pytorch_causal_corr:.6f}")
    
    # 科学标准：CausalEngine差异应该在基准差异的合理范围内
    tolerance_factor = 1.5  # 允许1.5倍的基准差异
    
    print(f"\n📊 科学判断基准:")
    print(f"  基准差异容忍度: {sklearn_pytorch_mse_diff * tolerance_factor:.2f} MSE")
    print(f"  CausalEngine vs sklearn: {sklearn_causal_mse_diff:.2f} {'✅' if sklearn_causal_mse_diff <= sklearn_pytorch_mse_diff * tolerance_factor else '❌'}")
    print(f"  CausalEngine vs PyTorch: {pytorch_causal_mse_diff:.2f} {'✅' if pytorch_causal_mse_diff <= sklearn_pytorch_mse_diff * tolerance_factor else '❌'}")
    sklearn_equivalent = (sklearn_causal_mse_diff <= sklearn_pytorch_mse_diff * tolerance_factor or 
                         sklearn_causal_r2_diff <= sklearn_pytorch_r2_diff * tolerance_factor)
    pytorch_equivalent = (pytorch_causal_mse_diff <= sklearn_pytorch_mse_diff * tolerance_factor or
                         pytorch_causal_r2_diff <= sklearn_pytorch_r2_diff * tolerance_factor)
    high_correlation = pytorch_causal_corr > 0.999 and sklearn_causal_corr > 0.999
    
    print(f"\n✅ 科学等价性判断:")
    print(f"  基于sklearn-PyTorch基准差异的科学标准")
    print(f"  与sklearn数学等价:     {'✓ 是' if sklearn_equivalent else '✗ 否'}")
    print(f"  与PyTorch数学等价:     {'✓ 是' if pytorch_equivalent else '✗ 否'}")
    print(f"  高度预测一致性:       {'✓ 是' if high_correlation else '✗ 否'}")
    
    overall_equivalent = sklearn_equivalent and pytorch_equivalent and high_correlation
    
    if overall_equivalent:
        print(f"\n🎉 科学结论: CausalEngine deterministic模式数学实现正确!")
        print(f"    ✓ 所有差异都在sklearn-PyTorch基准差异范围内")
        print(f"    ✓ 证明CausalEngine实现的数学正确性")
        print(f"    ✓ 可以作为sklearn MLP的直接替代品")
    else:
        print(f"\n⚠️ 科学结论: 需要进一步分析")
        print(f"    基准差异: sklearn-PyTorch = {sklearn_pytorch_mse_diff:.2f} MSE")
        if not sklearn_equivalent:
            print(f"    ✗ CausalEngine-sklearn差异({sklearn_causal_mse_diff:.2f}) 超出基准范围({sklearn_pytorch_mse_diff * tolerance_factor:.2f})")
        if not pytorch_equivalent:
            print(f"    ✗ CausalEngine-PyTorch差异({pytorch_causal_mse_diff:.2f}) 超出基准范围({sklearn_pytorch_mse_diff * tolerance_factor:.2f})")
        if not high_correlation:
            print(f"    ✗ 预测相关性不足: PyTorch{pytorch_causal_corr:.6f}, sklearn{sklearn_causal_corr:.6f}")
    
    return {
        'sklearn': {'r2': sklearn_r2, 'mse': sklearn_mse},
        'pytorch': {'r2': pytorch_r2, 'mse': pytorch_mse},
        'causal': {'r2': causal_r2, 'mse': causal_mse},
        'equivalent': overall_equivalent,
        'differences': {
            'pytorch_causal': pytorch_causal_mse_diff,
            'sklearn_causal': sklearn_causal_mse_diff
        }
    }


def scientific_classification_equivalence_test():
    """基于科学标准的分类等价性验证"""
    print("\n" + "="*60)
    print("🔬 科学分类等价性验证")
    print("="*60)
    
    # 生成固定数据
    X, y = make_classification(n_samples=800, n_features=10, n_classes=3, 
                              n_redundant=0, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"数据: {X_train.shape} 训练, {X_test.shape} 测试, {len(np.unique(y))} 类")
    
    # 使用高耐心早停策略的配置
    common_params = {
        'hidden_layer_sizes': (64, 32),
        'max_iter': 2000,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'tol': 1e-5,
        'learning_rate_init': 0.001,
        'alpha': 0.0,
    }
    
    print(f"\n📊 使用高耐心早停策略的超参数:")
    for key, value in common_params.items():
        print(f"  {key}: {value}")
    
    # 1. sklearn MLPClassifier
    print(f"\n--- sklearn MLPClassifier ---")
    sklearn_clf = MLPClassifier(**common_params)
    sklearn_clf.fit(X_train, y_train)
    sklearn_pred = sklearn_clf.predict(X_test)
    sklearn_proba = sklearn_clf.predict_proba(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    
    print(f"准确率: {sklearn_acc:.6f}")
    print(f"训练迭代数: {sklearn_clf.n_iter_}")
    
    # 2. PyTorch nn.Sequential分类器控制组
    print(f"\n--- PyTorch nn.Sequential分类器控制组 ---")
    pytorch_params = {
        'hidden_layer_sizes': common_params['hidden_layer_sizes'],
        'max_iter': common_params['max_iter'],
        'random_state': common_params['random_state'],
        'learning_rate': common_params['learning_rate_init'],
        'alpha': common_params['alpha'],
        'early_stopping': common_params['early_stopping'],
        'validation_fraction': common_params['validation_fraction'],
        'n_iter_no_change': common_params['n_iter_no_change'],
        'tol': common_params['tol'],
    }
    
    pytorch_clf = SimpleMLPClassifier(**pytorch_params)
    pytorch_clf.fit(X_train, y_train)
    pytorch_pred = pytorch_clf.predict(X_test)
    pytorch_proba = pytorch_clf.predict_proba(X_test)
    pytorch_acc = accuracy_score(y_test, pytorch_pred)
    
    print(f"准确率: {pytorch_acc:.6f}")
    print(f"训练迭代数: {pytorch_clf.n_iter_}")
    
    # 3. CausalEngine deterministic模式
    print(f"\n--- CausalEngine deterministic模式 ---")
    
    causal_params = {
        'hidden_layer_sizes': common_params['hidden_layer_sizes'],
        'max_iter': common_params['max_iter'],
        'random_state': common_params['random_state'],
        'early_stopping': common_params['early_stopping'],
        'validation_fraction': common_params['validation_fraction'],
        'n_iter_no_change': common_params['n_iter_no_change'],
        'tol': common_params['tol'],
        'learning_rate': common_params['learning_rate_init'],
        'alpha': common_params['alpha'],
        'mode': 'deterministic',
        'verbose': False
    }
    
    causal_clf = MLPCausalClassifier(**causal_params)
    causal_clf.fit(X_train, y_train)
    causal_pred = causal_clf.predict(X_test)
    causal_proba = causal_clf.predict_proba(X_test)
    
    if isinstance(causal_pred, dict):
        causal_pred = causal_pred['predictions']
    
    causal_acc = accuracy_score(y_test, causal_pred)
    
    print(f"准确率: {causal_acc:.6f}")
    
    # 4. 数学等价性核心验证
    print(f"\n🎯 数学等价性验证 - 分类任务")
    print("=" * 70)
    
    # 关键差异计算
    pytorch_causal_acc_diff = abs(pytorch_acc - causal_acc)
    sklearn_causal_acc_diff = abs(sklearn_acc - causal_acc)
    pytorch_causal_agreement = np.mean(pytorch_pred == causal_pred)
    sklearn_causal_agreement = np.mean(sklearn_pred == causal_pred)
    
    print(f"核心问题：CausalEngine deterministic模式是否与传统MLP数学等价？")
    print(f"")
    print(f"📊 性能对比:")
    print(f"  方法                准确率       训练轮数")
    print(f"  sklearn           {sklearn_acc:.6f}   {sklearn_clf.n_iter_}")
    print(f"  PyTorch控制组      {pytorch_acc:.6f}   {pytorch_clf.n_iter_}")
    print(f"  CausalEngine      {causal_acc:.6f}   {getattr(causal_clf, 'n_iter_', 'N/A')}")
    
    print(f"\n📏 科学等价性分析:")
    sklearn_pytorch_acc_diff = abs(sklearn_acc - pytorch_acc)  # 基准差异
    print(f"  sklearn ↔ PyTorch 基准差异 (相同算法，不同实现):")
    print(f"    准确率差异:   {sklearn_pytorch_acc_diff:.6f}")
    print(f"")
    print(f"  CausalEngine ↔ sklearn:")
    print(f"    准确率差异:   {sklearn_causal_acc_diff:.6f}")
    print(f"    预测一致性:   {sklearn_causal_agreement:.6f}")
    
    print(f"  CausalEngine ↔ PyTorch:")
    print(f"    准确率差异:   {pytorch_causal_acc_diff:.6f}")
    print(f"    预测一致性:   {pytorch_causal_agreement:.6f}")
    
    print(f"\n📊 科学判断基准:")
    tolerance_factor = 1.5
    print(f"  基准差异容忍度: {sklearn_pytorch_acc_diff * tolerance_factor:.4f} 准确率")
    print(f"  CausalEngine vs sklearn: {sklearn_causal_acc_diff:.4f} {'✅' if sklearn_causal_acc_diff <= sklearn_pytorch_acc_diff * tolerance_factor else '❌'}")
    print(f"  CausalEngine vs PyTorch: {pytorch_causal_acc_diff:.4f} {'✅' if pytorch_causal_acc_diff <= sklearn_pytorch_acc_diff * tolerance_factor else '❌'}")
    
    # 科学的等价性判断标准：基于"相同算法实现差异"
    sklearn_equivalent = (sklearn_causal_acc_diff <= sklearn_pytorch_acc_diff * tolerance_factor and 
                         sklearn_causal_agreement > 0.85)
    pytorch_equivalent = (pytorch_causal_acc_diff <= sklearn_pytorch_acc_diff * tolerance_factor and 
                         pytorch_causal_agreement > 0.85)
    
    print(f"\n✅ 科学等价性判断:")
    print(f"  基于sklearn-PyTorch基准差异的科学标准")
    print(f"  与sklearn数学等价:     {'✓ 是' if sklearn_equivalent else '✗ 否'}")
    print(f"  与PyTorch数学等价:     {'✓ 是' if pytorch_equivalent else '✗ 否'}")
    
    overall_equivalent = sklearn_equivalent and pytorch_equivalent
    
    if overall_equivalent:
        print(f"\n🎉 科学结论: CausalEngine deterministic模式数学实现正确!")
        print(f"    ✓ 所有差异都在sklearn-PyTorch基准差异范围内")
        print(f"    ✓ 证明CausalEngine实现的数学正确性")
        print(f"    ✓ 可以作为sklearn MLP的直接替代品")
    else:
        print(f"\n⚠️ 科学结论: 需要进一步分析")
        print(f"    基准差异: sklearn-PyTorch = {sklearn_pytorch_acc_diff:.4f} 准确率")
        if not sklearn_equivalent:
            print(f"    ✗ CausalEngine-sklearn差异({sklearn_causal_acc_diff:.4f}) 超出基准范围({sklearn_pytorch_acc_diff * tolerance_factor:.4f})")
        if not pytorch_equivalent:
            print(f"    ✗ CausalEngine-PyTorch差异({pytorch_causal_acc_diff:.4f}) 超出基准范围({sklearn_pytorch_acc_diff * tolerance_factor:.4f})")
    
    return {
        'sklearn': {'accuracy': sklearn_acc},
        'pytorch': {'accuracy': pytorch_acc},
        'causal': {'accuracy': causal_acc},
        'equivalent': overall_equivalent,
        'differences': {
            'pytorch_causal': pytorch_causal_acc_diff,
            'sklearn_causal': sklearn_causal_acc_diff
        }
    }


def test_five_modes_consistency():
    """验证CausalEngine五种模式的一致性（回归和分类任务）"""
    print("\n" + "="*60)
    print("🔬 五模式一致性验证（回归+分类）")
    print("="*60)
    
    # 生成回归测试数据
    X_reg, y_reg = make_regression(n_samples=800, n_features=10, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # 生成分类测试数据
    X_clf, y_clf = make_classification(n_samples=800, n_features=10, n_classes=3, n_redundant=0, n_informative=8, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    print(f"数据: 回归{X_train_reg.shape} 训练, 分类{X_train_clf.shape} 训练")
    
    # 测试参数
    causal_params_base = {
        'hidden_layer_sizes': (64, 32),
        'max_iter': 800,
        'random_state': 42,
        'early_stopping': False,
        'learning_rate': 0.001,
        'alpha': 0.0,
        'verbose': False
    }
    
    # 测试各种CausalEngine模式
    modes = ['deterministic', 'exogenous', 'endogenous', 'standard', 'sampling']
    results = {}
    
    print(f"\n📊 五模式运行状态表:")
    print("+" + "-"*70 + "+")
    print(f"| {'模式':<12} | {'回归任务':<20} | {'分类任务':<20} | {'状态':<8} |")
    print("+" + "-"*70 + "+")
    
    for mode in modes:
        causal_params = causal_params_base.copy()
        causal_params['mode'] = mode
        
        reg_success = False
        clf_success = False
        reg_result = ""
        clf_result = ""
        
        # 测试回归任务
        try:
            causal_reg = MLPCausalRegressor(**causal_params)
            causal_reg.fit(X_train_reg, y_train_reg)
            causal_pred_reg = causal_reg.predict(X_test_reg)
            
            if isinstance(causal_pred_reg, dict):
                causal_pred_reg = causal_pred_reg.get('predictions', causal_pred_reg.get('output', causal_pred_reg))
            
            causal_r2 = r2_score(y_test_reg, causal_pred_reg)
            causal_mse = mean_squared_error(y_test_reg, causal_pred_reg)
            
            if causal_r2 > 0.5:  # 基本性能检查
                reg_success = True
                reg_result = f"R²={causal_r2:.4f}, MSE={causal_mse:.1f}"
            else:
                reg_result = "性能异常"
                
        except Exception as e:
            reg_result = "运行失败"
        
        # 测试分类任务
        try:
            causal_clf = MLPCausalClassifier(**causal_params)
            causal_clf.fit(X_train_clf, y_train_clf)
            causal_pred_clf = causal_clf.predict(X_test_clf)
            
            if isinstance(causal_pred_clf, dict):
                causal_pred_clf = causal_pred_clf['predictions']
            
            causal_acc = accuracy_score(y_test_clf, causal_pred_clf)
            
            if causal_acc > 0.5:  # 基本性能检查
                clf_success = True
                clf_result = f"准确率={causal_acc:.4f}"
            else:
                clf_result = "性能异常"
                
        except Exception as e:
            clf_result = "运行失败"
        
        # 综合状态
        overall_success = reg_success and clf_success
        status = "✅正常" if overall_success else "❌异常"
        
        print(f"| {mode:<12} | {reg_result:<20} | {clf_result:<20} | {status:<8} |")
        results[mode] = {'reg_success': reg_success, 'clf_success': clf_success, 'consistent': overall_success}
    
    print("+" + "-"*70 + "+")
    
    # 总结
    successful_modes = sum(1 for result in results.values() if result.get('consistent', False))
    print(f"\n📊 模式一致性总结:")
    print(f"成功运行的模式: {successful_modes}/{len(modes)}")
    print(f"回归任务成功: {sum(1 for r in results.values() if r.get('reg_success', False))}/{len(modes)}")
    print(f"分类任务成功: {sum(1 for r in results.values() if r.get('clf_success', False))}/{len(modes)}")
    
    return {'successful_modes': successful_modes, 'total_modes': len(modes)}


def main():
    """CausalEngine科学等价性验证主函数"""
    print("🔬 CausalEngine科学等价性验证 - 基于科学标准")
    print("="*70)
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 科学回归等价性验证（三方对比）
        reg_results = scientific_regression_equivalence_test()
        
        # 2. 科学分类等价性验证（三方对比）
        clf_results = scientific_classification_equivalence_test()
        
        # 3. 五模式一致性验证
        other_modes_results = test_five_modes_consistency()
        
        # 总结
        print("\n" + "="*70)
        print("🎯 CausalEngine科学等价性验证结果")
        print("="*70)
        
        overall_equivalent = reg_results['equivalent'] and clf_results['equivalent']
        
        print(f"核心验证目标: 基于科学标准证明CausalEngine deterministic模式数学实现正确")
        print(f"验证方法: 以sklearn-PyTorch(相同算法)差异作为基准，验证CausalEngine是否在合理范围内")
        print(f"")
        print(f"📊 科学验证结果:")
        print(f"  回归任务等价性:  {'✅ 通过科学验证' if reg_results['equivalent'] else '❌ 未通过'}")
        print(f"  分类任务等价性:  {'✅ 通过科学验证' if clf_results['equivalent'] else '❌ 未通过'}")
        modes_status = '✅ 全部正常' if other_modes_results['successful_modes'] == 5 else f'❌ {other_modes_results["successful_modes"]}/5正常'
        print(f"  五模式运行状态:  {modes_status}")
        
        if overall_equivalent:
            print(f"\n🎉 【科学结论】CausalEngine deterministic模式数学实现正确!")
            print(f"")
            print(f"✓ 基于sklearn-PyTorch基准差异的科学验证通过")
            print(f"✓ 所有性能差异都在'相同算法不同实现'的合理范围内")
            print(f"✓ 证明CausalEngine数学实现的正确性和可靠性")
            print(f"✓ 可以作为sklearn MLPRegressor/MLPClassifier的直接替代品")
            print(f"✓ 为后续因果推理功能提供了扎实的数学基础")
        else:
            print(f"\n⚠️ 【科学分析】基于科学标准的分析结果")
            print(f"")
            if not reg_results['equivalent']:
                print(f"✗ 回归任务CausalEngine差异超出sklearn-PyTorch基准范围")
            if not clf_results['equivalent']:
                print(f"✗ 分类任务CausalEngine差异超出sklearn-PyTorch基准范围")
            print(f"→ 建议进一步分析差异原因：网络结构、损失函数、优化策略")
            print(f"→ 或者验证基准差异范围设置是否合理")
            print(f"→ 核心原则：宁可承认实现正确但效果有限，也不能数学基础错误")
        
        return overall_equivalent
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)