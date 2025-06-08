"""
柯西分布工具函数

实现柯西分布的核心数学运算，包括概率密度函数、累积分布函数、
分位数函数以及线性组合等关键操作。
"""

import torch
import torch.nn as nn
import math


def cauchy_pdf(x, loc, scale):
    """
    计算柯西分布的概率密度函数
    
    Args:
        x: 输入值
        loc: 位置参数 μ
        scale: 尺度参数 γ (必须 > 0)
    
    Returns:
        概率密度值
    """
    return 1.0 / (math.pi * scale * (1 + ((x - loc) / scale) ** 2))


def cauchy_cdf(x, loc, scale):
    """
    计算柯西分布的累积分布函数
    
    Args:
        x: 输入值
        loc: 位置参数 μ
        scale: 尺度参数 γ
    
    Returns:
        累积概率值
    """
    return 0.5 + (1.0 / math.pi) * torch.atan((x - loc) / scale)


def cauchy_quantile(p, loc, scale):
    """
    计算柯西分布的分位数函数（CDF的反函数）
    
    Args:
        p: 概率值 (0 < p < 1)
        loc: 位置参数 μ
        scale: 尺度参数 γ
    
    Returns:
        对应的分位数
    """
    return loc + scale * torch.tan(math.pi * (p - 0.5))


def cauchy_sample(loc, scale, size=None):
    """
    从柯西分布中采样
    
    Args:
        loc: 位置参数 μ
        scale: 尺度参数 γ
        size: 采样数量
    
    Returns:
        采样值
    """
    if size is None:
        u = torch.rand_like(loc)
    else:
        u = torch.rand(size, device=loc.device, dtype=loc.dtype)
    
    return cauchy_quantile(u, loc, scale)


def cauchy_linear_transform(input_loc, input_scale, weight, bias=None):
    """
    计算柯西分布线性变换的结果分布参数
    
    基于柯西分布的重要性质：独立柯西随机变量的线性组合仍为柯西分布
    如果 X_i ~ Cauchy(μ_i, γ_i)，则 Y = Σ(w_i * X_i) + b ~ Cauchy(Σ(w_i * μ_i) + b, Σ(|w_i| * γ_i))
    
    Args:
        input_loc: 输入分布的位置参数 [batch_size, input_dim]
        input_scale: 输入分布的尺度参数 [batch_size, input_dim]
        weight: 线性变换权重 [output_dim, input_dim]
        bias: 偏置项 [output_dim] (可选)
    
    Returns:
        (output_loc, output_scale): 输出分布的参数
    """
    # 计算输出位置参数：Σ(w_i * μ_i) + b
    output_loc = torch.matmul(input_loc, weight.t())
    if bias is not None:
        output_loc = output_loc + bias
    
    # 计算输出尺度参数：Σ(|w_i| * γ_i)
    output_scale = torch.matmul(input_scale, torch.abs(weight).t())
    
    return output_loc, output_scale


class CauchyLinear(nn.Module):
    """
    柯西分布的线性变换层
    
    实现从输入柯西分布到输出柯西分布的线性变换，
    保持分布的柯西性质。
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_loc, input_scale):
        """
        前向传播
        
        Args:
            input_loc: 输入分布位置参数 [batch_size, in_features]
            input_scale: 输入分布尺度参数 [batch_size, in_features]
        
        Returns:
            (output_loc, output_scale): 输出分布参数
        """
        return cauchy_linear_transform(input_loc, input_scale, self.weight, self.bias)


def threshold_probability(loc, scale, threshold=0.0):
    """
    计算柯西分布超过阈值的概率
    
    用于OvR分类中计算每个类别的预测概率
    P(X > threshold) = 1 - CDF(threshold)
    
    Args:
        loc: 位置参数
        scale: 尺度参数  
        threshold: 阈值
    
    Returns:
        超过阈值的概率
    """
    return 1.0 - cauchy_cdf(threshold, loc, scale)


def log_cauchy_pdf(x, loc, scale):
    """
    计算柯西分布的对数概率密度
    
    用于数值稳定的概率计算
    """
    return -torch.log(math.pi * scale) - torch.log(1 + ((x - loc) / scale) ** 2)

