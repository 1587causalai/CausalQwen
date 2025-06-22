"""
传统神经网络基准算法
用于与CausalEngine进行消融对比实验的基准模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BaselineMLPClassifier(nn.Module):
    """
    基准MLP分类器 - 新命名约定
    等价于TraditionalMLPClassifier，提供统一的接口
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int, 
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 构建网络层
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率分布"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


class TraditionalMLPClassifier(nn.Module):
    """
    传统多层感知机分类器
    
    这是标准的前馈神经网络实现，用作CausalEngine消融实验的基准。
    网络结构等价于CausalEngine仅使用位置输出(loc)时的行为。
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
            
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_size]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测概率分布"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)


class BaselineMLPRegressor(nn.Module):
    """
    基准MLP回归器 - 新命名约定
    等价于TraditionalMLPRegressor，提供统一的接口
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建网络层
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(), 
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)
    
    def predict(self, x: torch.Tensor, temperature=None, do_sample=None) -> torch.Tensor:
        """预测输出 - 兼容CausalEngine API"""
        with torch.no_grad():
            return self.forward(x)


class TraditionalMLPRegressor(nn.Module):
    """
    传统多层感知机回归器
    
    这是标准的前馈神经网络回归实现，用作CausalEngine消融实验的基准。
    网络结构等价于CausalEngine仅使用位置输出(loc)时的行为。
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_sizes: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
            
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        # 输出激活函数（可选）
        if output_activation is not None:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "none": nn.Identity()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_size]
            
        Returns:
            output: 回归预测值 [batch_size, output_size]
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor, temperature=None, do_sample=None) -> torch.Tensor:
        """预测输出 - 兼容CausalEngine API"""
        with torch.no_grad():
            return self.forward(x)


class TraditionalCNNClassifier(nn.Module):
    """
    传统卷积神经网络分类器
    用于图像或序列分类任务的基准模型
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_layers: Optional[List[int]] = None,
        fc_layers: Optional[List[int]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        if conv_layers is None:
            conv_layers = [32, 64, 128]
        if fc_layers is None:
            fc_layers = [256, 128]
            
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 卷积层
        conv_modules = []
        in_channels = input_channels
        
        for out_channels in conv_layers:
            conv_modules.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_modules)
        
        # 全连接层
        fc_modules = []
        # 这里需要根据实际输入大小动态计算
        # 为简化，假设经过卷积后的大小
        prev_size = conv_layers[-1]  # 需要根据实际情况调整
        
        for hidden_size in fc_layers:
            fc_modules.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
            
        fc_modules.append(nn.Linear(prev_size, num_classes))
        self.fc_layers = nn.Sequential(*fc_modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, channels, length] 
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool1d(x, 1)  # 全局平均池化
        x = x.squeeze(-1)  # 移除最后一个维度
        x = self.fc_layers(x)
        return x


def create_baseline_classifier(
    input_size: int,
    num_classes: int,
    architecture: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    创建基准分类器的工厂函数
    
    Args:
        input_size: 输入特征维度
        num_classes: 类别数
        architecture: 网络架构类型 ("mlp", "cnn")
        **kwargs: 其他网络参数
        
    Returns:
        baseline_model: 基准分类模型
    """
    if architecture == "mlp":
        return TraditionalMLPClassifier(input_size, num_classes, **kwargs)
    elif architecture == "cnn":
        return TraditionalCNNClassifier(input_size, num_classes, **kwargs)
    else:
        raise ValueError(f"不支持的架构类型: {architecture}")


def create_baseline_regressor(
    input_size: int,
    output_size: int = 1,
    architecture: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    创建基准回归器的工厂函数
    
    Args:
        input_size: 输入特征维度
        output_size: 输出维度
        architecture: 网络架构类型
        **kwargs: 其他网络参数
        
    Returns:
        baseline_model: 基准回归模型
    """
    if architecture == "mlp":
        return TraditionalMLPRegressor(input_size, output_size, **kwargs)
    else:
        raise ValueError(f"不支持的架构类型: {architecture}")


# 训练器类
class BaselineTrainer:
    """
    基准模型训练器
    提供统一的训练接口用于消融实验
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    def train_classification(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        """训练分类模型"""
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        return self.model
    
    def train_regression(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        """训练回归模型"""
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # 修复维度不匹配：如果输出是[batch_size, 1]，squeeze到[batch_size]
                if outputs.dim() > 1 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    
                    # 修复维度不匹配：如果输出是[batch_size, 1]，squeeze到[batch_size]
                    if outputs.dim() > 1 and outputs.size(-1) == 1:
                        outputs = outputs.squeeze(-1)
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        return self.model