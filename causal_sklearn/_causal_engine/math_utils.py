"""
Cauchy Distribution Mathematical Utilities for CausalEngine

柯西分布数学工具，实现线性稳定性和相关计算。
这些工具是CausalEngine解析计算能力的核心。
"""

import torch
import numpy as np
from typing import Union, Tuple


class CauchyMath:
    """
    柯西分布数学工具类
    
    实现柯西分布的核心数学性质：
    1. 概率密度函数 (PDF)
    2. 累积分布函数 (CDF)  
    3. 线性稳定性计算
    4. 负对数似然 (NLL) 损失
    
    数学基础：
    - PDF: f(x) = 1/(π·γ[1 + ((x-μ)/γ)²])
    - CDF: F(x) = 1/2 + (1/π)arctan((x-μ)/γ)
    - 线性稳定性: aX + b ~ Cauchy(aμ + b, |a|γ)
    """
    
    @staticmethod
    def pdf(
        x: torch.Tensor, 
        loc: torch.Tensor, 
        scale: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        柯西分布概率密度函数
        
        Args:
            x: 输入值 [batch_size, ...]
            loc: 位置参数 μ [batch_size, ...]
            scale: 尺度参数 γ [batch_size, ...] (必须 > 0)
            eps: 数值稳定性常数
            
        Returns:
            pdf: 概率密度值 [batch_size, ...]
        """
        # 标准化：z = (x - μ) / γ
        z = (x - loc) / (scale + eps)
        
        # PDF: f(x) = 1/(π·γ) * 1/(1 + z²)
        pdf = 1.0 / (torch.pi * (scale + eps) * (1.0 + z * z))
        
        return pdf
    
    @staticmethod
    def cdf(
        x: torch.Tensor, 
        loc: torch.Tensor, 
        scale: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        柯西分布累积分布函数
        
        Args:
            x: 输入值 [batch_size, ...]
            loc: 位置参数 μ [batch_size, ...]  
            scale: 尺度参数 γ [batch_size, ...] (必须 > 0)
            eps: 数值稳定性常数
            
        Returns:
            cdf: 累积概率值 [batch_size, ...] ∈ [0, 1]
        """
        # 标准化：z = (x - μ) / γ
        z = (x - loc) / (scale + eps)
        
        # CDF: F(x) = 1/2 + (1/π)arctan(z)
        cdf = 0.5 + (1.0 / torch.pi) * torch.atan(z)
        
        return cdf
    
    @staticmethod
    def survival_function(
        x: torch.Tensor, 
        loc: torch.Tensor, 
        scale: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        柯西分布生存函数 (1 - CDF)
        
        用于分类任务：P(S_k > threshold) = 1 - F(threshold)
        
        Args:
            x: 输入值 [batch_size, ...]
            loc: 位置参数 μ [batch_size, ...]
            scale: 尺度参数 γ [batch_size, ...] (必须 > 0)  
            eps: 数值稳定性常数
            
        Returns:
            survival: 生存概率 P(X > x) [batch_size, ...] ∈ [0, 1]
        """
        return 1.0 - CauchyMath.cdf(x, loc, scale, eps)
    
    @staticmethod
    def log_prob(
        x: torch.Tensor, 
        loc: torch.Tensor, 
        scale: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        柯西分布对数概率密度
        
        用于计算负对数似然损失
        
        Args:
            x: 输入值 [batch_size, ...]
            loc: 位置参数 μ [batch_size, ...]
            scale: 尺度参数 γ [batch_size, ...] (必须 > 0)
            eps: 数值稳定性常数
            
        Returns:
            log_prob: 对数概率密度 [batch_size, ...]
        """
        # 标准化：z = (x - μ) / γ
        z = (x - loc) / (scale + eps)
        
        # log PDF: log f(x) = -log(π) - log(γ) - log(1 + z²)
        log_prob = -torch.log(torch.tensor(torch.pi)) - torch.log(scale + eps) - torch.log(1.0 + z * z)
        
        return log_prob
    
    @staticmethod
    def nll_loss(
        y_true: torch.Tensor, 
        loc: torch.Tensor, 
        scale: torch.Tensor,
        reduction: str = 'mean',
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        柯西分布负对数似然损失
        
        用于回归任务的因果模式损失函数
        
        Args:
            y_true: 真实值 [batch_size, ...]
            loc: 预测位置参数 [batch_size, ...]
            scale: 预测尺度参数 [batch_size, ...] (必须 > 0)
            reduction: 损失归约方式 ('mean', 'sum', 'none')
            eps: 数值稳定性常数
            
        Returns:
            loss: 负对数似然损失
        """
        # NLL = -log P(y|μ,γ)
        log_prob = CauchyMath.log_prob(y_true, loc, scale, eps)
        nll = -log_prob
        
        if reduction == 'mean':
            return torch.mean(nll)
        elif reduction == 'sum':
            return torch.sum(nll)
        elif reduction == 'none':
            return nll
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    @staticmethod
    def linear_transform(
        loc: torch.Tensor, 
        scale: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        柯西分布的线性变换
        
        如果 X ~ Cauchy(μ, γ)，则 W·X + b ~ Cauchy(W·μ + b, |W|·γ)
        
        这是CausalEngine解析计算的核心数学性质
        
        Args:
            loc: 输入位置参数 [batch_size, input_dim]
            scale: 输入尺度参数 [batch_size, input_dim] 
            weight: 权重矩阵 [output_dim, input_dim]
            bias: 偏置向量 [output_dim]
            
        Returns:
            new_loc: 变换后位置参数 [batch_size, output_dim]
            new_scale: 变换后尺度参数 [batch_size, output_dim]
        """
        # 位置参数的线性变换: μ' = W·μ + b
        new_loc = torch.matmul(loc, weight.T) + bias
        
        # 尺度参数的变换: γ' = |W|·γ
        new_scale = torch.matmul(scale, torch.abs(weight).T)
        
        return new_loc, new_scale
    
    @staticmethod
    def add_distributions(
        loc1: torch.Tensor, 
        scale1: torch.Tensor,
        loc2: torch.Tensor, 
        scale2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        柯西分布的加法稳定性
        
        如果 X1 ~ Cauchy(μ1, γ1), X2 ~ Cauchy(μ2, γ2) 且独立，
        则 X1 + X2 ~ Cauchy(μ1 + μ2, γ1 + γ2)
        
        Args:
            loc1, scale1: 第一个分布的参数
            loc2, scale2: 第二个分布的参数
            
        Returns:
            sum_loc: 和分布的位置参数
            sum_scale: 和分布的尺度参数
        """
        sum_loc = loc1 + loc2
        sum_scale = scale1 + scale2
        
        return sum_loc, sum_scale
    
    @staticmethod
    def sample(
        loc: torch.Tensor, 
        scale: torch.Tensor, 
        sample_shape: Tuple[int, ...] = ()
    ) -> torch.Tensor:
        """
        从柯西分布采样
        
        使用逆变换采样方法：X = μ + γ * tan(π(U - 0.5))
        其中 U ~ Uniform(0, 1)
        
        Args:
            loc: 位置参数 [batch_size, ...]
            scale: 尺度参数 [batch_size, ...]
            sample_shape: 额外的采样维度
            
        Returns:
            samples: 采样结果 [sample_shape..., batch_size, ...]
        """
        # 生成均匀分布采样
        shape = sample_shape + loc.shape
        uniform = torch.rand(shape, dtype=loc.dtype, device=loc.device)
        
        # 逆变换采样: X = μ + γ * tan(π(U - 0.5))
        samples = loc + scale * torch.tan(torch.pi * (uniform - 0.5))
        
        return samples


# Numpy版本的工具函数，用于sklearn兼容性
class CauchyMathNumpy:
    """
    柯西分布数学工具的NumPy版本
    
    用于与sklearn的兼容性，不依赖PyTorch
    """
    
    @staticmethod
    def pdf(x: np.ndarray, loc: np.ndarray, scale: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """柯西分布PDF (NumPy版本)"""
        z = (x - loc) / (scale + eps)
        return 1.0 / (np.pi * (scale + eps) * (1.0 + z * z))
    
    @staticmethod
    def cdf(x: np.ndarray, loc: np.ndarray, scale: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """柯西分布CDF (NumPy版本)"""
        z = (x - loc) / (scale + eps)
        return 0.5 + (1.0 / np.pi) * np.arctan(z)
    
    @staticmethod
    def nll_loss(
        y_true: np.ndarray, 
        loc: np.ndarray, 
        scale: np.ndarray,
        eps: float = 1e-8
    ) -> float:
        """柯西分布NLL损失 (NumPy版本)"""
        z = (y_true - loc) / (scale + eps)
        nll = np.log(np.pi) + np.log(scale + eps) + np.log(1.0 + z * z)
        return np.mean(nll)
    
    @staticmethod
    def sample(
        loc: np.ndarray, 
        scale: np.ndarray, 
        size: Tuple[int, ...] = None
    ) -> np.ndarray:
        """从柯西分布采样 (NumPy版本)"""
        if size is None:
            size = loc.shape
        
        uniform = np.random.rand(*size)
        return loc + scale * np.tan(np.pi * (uniform - 0.5))