"""
CausalEngine Activation Heads

激活头是 CausalEngine 与世界交互的最终接口。
它将原始的因果决策分布 S ~ Cauchy(loc_S, scale_S) 转换为可用的输出：
- 分类激活：计算 P(S_k > C_k)
- 回归激活：计算 a_k * S_k + b_k

这种设计实现了因果核心与决策应用的完美解耦。
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict
from enum import Enum


class ActivationMode(Enum):
    """激活模式枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ActivationHead(nn.Module):
    """
    统一激活头：将因果决策转换为最终输出
    
    这是 CausalEngine 的最后一步，将抽象的因果决策分布转换为
    具体的预测结果。每个输出维度可以独立选择激活模式。
    
    Args:
        output_size: 输出维度数
        activation_modes: 每个维度的激活模式，可以是：
            - 单个模式字符串：所有维度使用相同模式
            - 模式列表：为每个维度指定模式
            - None：默认全部使用分类模式
        classification_threshold_init: 分类阈值初始值
        regression_scale_init: 回归缩放初始值
        regression_bias_init: 回归偏置初始值
    """
    
    def __init__(
        self,
        output_size: int,
        activation_modes: Optional[Union[str, List[str]]] = None,
        classification_threshold_init: float = 0.0,
        regression_scale_init: float = 1.0,
        regression_bias_init: float = 0.0
    ):
        super().__init__()
        
        self.output_size = output_size
        
        # 解析激活模式
        self.activation_modes = self._parse_activation_modes(activation_modes)
        
        # 统计各模式的维度
        self.classification_dims = [
            i for i, mode in enumerate(self.activation_modes) 
            if mode == ActivationMode.CLASSIFICATION
        ]
        self.regression_dims = [
            i for i, mode in enumerate(self.activation_modes)
            if mode == ActivationMode.REGRESSION
        ]
        
        # 分类参数：阈值 C_k
        if self.classification_dims:
            self.classification_thresholds = nn.Parameter(
                torch.full((len(self.classification_dims),), classification_threshold_init)
            )
        
        # 回归参数：缩放 a_k 和偏置 b_k
        if self.regression_dims:
            self.regression_scales = nn.Parameter(
                torch.full((len(self.regression_dims),), regression_scale_init)
            )
            self.regression_biases = nn.Parameter(
                torch.full((len(self.regression_dims),), regression_bias_init)
            )
    
    def _parse_activation_modes(self, modes: Optional[Union[str, List[str]]]) -> List[ActivationMode]:
        """解析激活模式配置"""
        if modes is None:
            # 默认全部分类
            return [ActivationMode.CLASSIFICATION] * self.output_size
        
        if isinstance(modes, str):
            # 单一模式应用到所有维度
            mode = ActivationMode(modes)
            return [mode] * self.output_size
        
        if isinstance(modes, list):
            # 为每个维度指定模式
            if len(modes) != self.output_size:
                raise ValueError(
                    f"激活模式列表长度 ({len(modes)}) 必须等于输出维度 ({self.output_size})"
                )
            return [ActivationMode(mode) for mode in modes]
        
        raise ValueError(f"不支持的激活模式类型: {type(modes)}")
    
    def forward(
        self,
        loc_S: torch.Tensor,
        scale_S: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播：应用激活函数
        
        Args:
            loc_S: [batch_size, seq_len, output_size] 决策位置参数
            scale_S: [batch_size, seq_len, output_size] 决策尺度参数
            return_dict: 是否返回字典格式
            
        Returns:
            如果 return_dict=True:
                包含 'output', 'classification_probs', 'regression_values' 的字典
            否则:
                output: [batch_size, seq_len, output_size] 混合输出
        """
        batch_size, seq_len, _ = loc_S.shape
        device = loc_S.device
        
        # 初始化输出张量
        output = torch.zeros(batch_size, seq_len, self.output_size, device=device)
        
        # 分类激活：P(S_k > C_k) = 1/2 + (1/π)arctan((loc_S_k - C_k)/scale_S_k)
        if self.classification_dims:
            # 提取分类维度
            loc_S_cls = loc_S[:, :, self.classification_dims]
            scale_S_cls = scale_S[:, :, self.classification_dims]
            
            # 计算概率
            normalized_diff = (loc_S_cls - self.classification_thresholds) / scale_S_cls
            probs = 0.5 + (1 / torch.pi) * torch.atan(normalized_diff)
            
            # 填充输出
            output[:, :, self.classification_dims] = probs
        
        # 回归激活：a_k * loc_S_k + b_k
        if self.regression_dims:
            # 提取回归维度
            loc_S_reg = loc_S[:, :, self.regression_dims]
            
            # 计算回归值
            values = self.regression_scales * loc_S_reg + self.regression_biases
            
            # 填充输出
            output[:, :, self.regression_dims] = values
        
        if return_dict:
            result = {
                'output': output,
                'activation_modes': self.activation_modes
            }
            
            # 添加分类和回归的详细输出
            if self.classification_dims:
                result['classification_probs'] = output[:, :, self.classification_dims]
                result['classification_dims'] = self.classification_dims
            
            if self.regression_dims:
                result['regression_values'] = output[:, :, self.regression_dims]
                result['regression_dims'] = self.regression_dims
            
            return result
        
        return output
    
    def get_config(self) -> Dict:
        """获取激活头配置"""
        config = {
            'output_size': self.output_size,
            'num_classification_dims': len(self.classification_dims),
            'num_regression_dims': len(self.regression_dims),
            'activation_modes': [mode.value for mode in self.activation_modes]
        }
        
        if self.classification_dims:
            config['classification_thresholds'] = self.classification_thresholds.tolist()
        
        if self.regression_dims:
            config['regression_scales'] = self.regression_scales.tolist()
            config['regression_biases'] = self.regression_biases.tolist()
        
        return config


class MultiTaskActivationHead(nn.Module):
    """
    多任务激活头：支持多个独立的激活头
    
    适用于需要多个不同输出的场景，例如：
    - 语言模型：词汇预测（分类）+ 情感分数（回归）
    - 多模态模型：图像分类 + 边界框回归
    
    Args:
        heads_config: 字典，键为任务名称，值为该任务的激活头配置
    """
    
    def __init__(self, heads_config: Dict[str, Dict]):
        super().__init__()
        
        self.heads = nn.ModuleDict()
        
        for task_name, config in heads_config.items():
            self.heads[task_name] = ActivationHead(**config)
    
    def forward(
        self,
        loc_S: Dict[str, torch.Tensor],
        scale_S: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播：对每个任务应用相应的激活头
        
        Args:
            loc_S: 字典，键为任务名称，值为该任务的位置参数
            scale_S: 字典，键为任务名称，值为该任务的尺度参数
            
        Returns:
            字典，键为任务名称，值为该任务的输出
        """
        outputs = {}
        
        for task_name, head in self.heads.items():
            if task_name not in loc_S or task_name not in scale_S:
                raise ValueError(f"缺少任务 '{task_name}' 的输入")
            
            outputs[task_name] = head(loc_S[task_name], scale_S[task_name])
        
        return outputs 