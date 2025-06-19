"""
CausalQwen MVP: 核心功能组件
包含：数学工具类、归因网络、行动网络、分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CausalQwen2Config


class CauchyMath:
    """Cauchy分布数学工具类，实现严格的线性稳定性"""
    
    @staticmethod
    def cauchy_linear_stable_loc(loc_input, weight, bias=None):
        """Cauchy分布位置参数的线性变换"""
        # 位置参数变换：直接矩阵乘法
        result = loc_input @ weight.T
        if bias is not None:
            result = result + bias
        return result
    
    @staticmethod  
    def cauchy_linear_stable_scale(scale_input, weight):
        """Cauchy分布尺度参数的线性变换"""
        # 尺度参数变换：直接矩阵乘法
        return scale_input @ torch.abs(weight).T


class AbductionNetwork(nn.Module):
    """归因网络：从隐藏状态推断个体表征分布"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        
        # 修正：添加bias项，符合设计文档要求
        self.loc_net = nn.Linear(config.hidden_size, config.causal_size, bias=True)
        self.scale_net = nn.Linear(config.hidden_size, config.causal_size, bias=True)
        
        self._init_identity_mapping()
    
    def _init_identity_mapping(self):
        """初始化为恒等映射，符合设计文档"""
        with torch.no_grad():
            if self.config.hidden_size == self.config.causal_size:
                # loc_net恒等映射初始化
                self.loc_net.weight.copy_(torch.eye(self.config.causal_size))
                self.loc_net.bias.zero_()
                
                # scale_net初始化：weight=0, bias=γ_init 产生宽分布
                self.scale_net.weight.zero_()
                self.scale_net.bias.fill_(self.config.gamma_init)
            else:
                # 如果维度不匹配，使用Xavier初始化
                nn.init.xavier_uniform_(self.loc_net.weight)
                nn.init.zeros_(self.loc_net.bias)
                nn.init.xavier_uniform_(self.scale_net.weight)
                self.scale_net.weight.data *= 0.1
                self.scale_net.bias.fill_(self.config.gamma_init)
    
    def forward(self, hidden_states):
        """前向传播，符合设计文档的数学要求"""
        # 位置参数：标准线性变换
        loc_U = self.loc_net(hidden_states)
        
        # 尺度参数：使用softplus确保正性，符合设计文档
        scale_U = F.softplus(self.scale_net(hidden_states))
        
        return loc_U, scale_U


class ActionNetwork(nn.Module):
    """行动网络：从个体表征到决策分布"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        
        # 修正：添加bias项，符合设计文档要求
        self.lm_head = nn.Linear(config.causal_size, config.vocab_size, bias=True)
        
        # 修正：b_noise维度应该是causal_size，用于外生噪声融合
        self.b_noise = nn.Parameter(torch.zeros(config.causal_size))
        
        self._init_from_original_lm_head()
    
    def _init_from_original_lm_head(self):
        """从原始lm_head复制权重，符合知识继承原则"""
        # 外生噪声应有合理的初始值，而非假设无噪声
        nn.init.constant_(self.b_noise, self.config.b_noise_init)
        
        # TODO: 当有预训练模型可用时，应从其复制权重
        # 目前使用标准初始化作为备选
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)
        
    def copy_weights_from_qwen(self, qwen_model):
        """从预训练Qwen2模型复制lm_head权重"""
        if hasattr(qwen_model, 'lm_head'):
            print("正在复制Qwen2预训练权重...")
            with torch.no_grad():
                # 确保vocab_size一致（包含预留词汇）
                if qwen_model.lm_head.weight.shape == self.lm_head.weight.shape:
                    self.lm_head.weight.copy_(qwen_model.lm_head.weight)
                    if hasattr(qwen_model.lm_head, 'bias') and qwen_model.lm_head.bias is not None:
                        self.lm_head.bias.copy_(qwen_model.lm_head.bias)
                    print(f"✅ 成功复制权重，词汇表大小: {qwen_model.lm_head.weight.shape[0]}")
                else:
                    print(f"❌ 权重形状不匹配: Qwen({qwen_model.lm_head.weight.shape}) vs CausalQwen({self.lm_head.weight.shape})")
                    print("使用标准初始化...")
        else:
            print("❌ 源模型没有lm_head，使用标准初始化...")
        
    def forward(self, loc_U, scale_U=None, do_sample=False, temperature=1.0):
        """前向传播 - 双模式设计：位置vs尺度的差异化处理
        
        核心机制：噪声对采样/非采样模式的不同影响方式
        
        采样模式：噪声影响位置参数
        ├─ ε ~ Cauchy(0, 1) 标准噪声采样
        ├─ U' ~ Cauchy(μ + T·|b_noise|·ε, γ) 噪声注入位置参数
        └─ 扰动个体身份，保持原有不确定性
        
        非采样模式：噪声影响尺度参数  
        ├─ U' ~ Cauchy(μ, γ + |b_noise|) 噪声融合到尺度
        └─ 保持个体身份，增加决策不确定性
        
        Args:
            loc_U: 个体表征分布的位置参数 [B, S, C]
            scale_U: 个体表征分布的尺度参数 [B, S, C]
            do_sample: 是否进行采样（决定噪声作用方式）
            temperature: 采样温度参数（仅在do_sample=True时生效）
        Returns:
            loc_S: 决策分布的位置参数 [B, S, V]
            scale_S: 决策分布的尺度参数 [B, S, V]
        """
        # 处理默认尺度参数
        if scale_U is None:
            scale_U = torch.zeros_like(loc_U)  # 默认为确定性分布
        
        if do_sample:
            # 🎯 采样模式：噪声影响位置参数
            
            # Step 1: 采样标准柯西噪声 ε ~ Cauchy(0, I)
            uniform_sample = torch.rand_like(loc_U)
            epsilon = torch.tan(torch.pi * (uniform_sample - 0.5))
            
            # Step 2: 温度调节的噪声注入到位置参数
            # 数学：loc_U_noisy = μ + T·|b_noise|·ε
            noise_injection = epsilon * temperature * torch.abs(self.b_noise)
            loc_U_noisy = loc_U + noise_injection
            
            # Step 3: 基于扰动后的位置参数进行线性决策
            # 数学：loc_S = W·(μ + T·|b_noise|·ε) + b
            loc_S = self.lm_head(loc_U_noisy)
            
            # Step 4: 尺度参数的线性稳定性变换
            # 数学：scale_S = γ × |W|^T
            scale_S = scale_U @ torch.abs(self.lm_head.weight).T

        else:
            # 🔧 非采样模式：噪声影响尺度参数
            
            # Step 1: 外生噪声融合到尺度参数
            # 数学：scale_U_noisy = γ + |b_noise|
            scale_U_noisy = scale_U + torch.abs(self.b_noise)
            
            # Step 2: 位置参数保持确定性的线性变换
            # 数学：loc_S = W·μ + b
            loc_S = self.lm_head(loc_U)
            
            # Step 3: 尺度参数的线性稳定性变换
            # 数学：scale_S = (γ + |b_noise|) × |W|^T
            scale_S = scale_U_noisy @ torch.abs(self.lm_head.weight).T
        
        return loc_S, scale_S


class OvRClassifier(nn.Module):
    """One-vs-Rest分类器"""
    
    def __init__(self, config: CausalQwen2Config):
        super().__init__()
        self.config = config
        self.thresholds = nn.Parameter(torch.full((config.vocab_size,), config.ovr_threshold_init))
    
    def forward(self, loc_S, scale_S):
        """计算OvR概率 - 占位实现"""
        # TODO: 实现严格的Cauchy分布CDF计算
        # 占位：使用简化的概率计算
        # P(y_k=1) = P(Cauchy(loc_S_k, scale_S_k) > threshold_k)
        normalized_diff = (loc_S - self.thresholds.unsqueeze(0).unsqueeze(0)) / scale_S
        # 使用atan近似Cauchy CDF: P = 0.5 + (1/π) * atan(x)
        probs = 0.5 + (1/torch.pi) * torch.atan(normalized_diff)
        return probs 