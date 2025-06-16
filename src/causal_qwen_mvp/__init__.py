"""
CausalQwen MVP: 因果语言模型最小可行产品

这是CausalQwen的MVP实现，使用占位式逻辑快速搭建基础框架。
核心特性：
- 继承Qwen2ForCausalLM，最大化复用现有基础设施
- 实现三种推理模式：标准、因果采样、兼容传统
- 支持完整的训练和验证流程
- 模块化设计，便于逐步完善

使用示例：
```python
from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config
from causal_qwen_mvp import CausalTrainer, CausalInferenceEngine

# 创建配置
config = CausalQwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B")

# 创建模型
model = CausalQwenMVPForCausalLM(config)

# 推理测试
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
output = model.inference(input_ids, mode='standard')
```
"""

from .models import (
    CausalQwenMVPForCausalLM,
    CausalQwen2Config,
    CausalMVPOutput,
    CauchyMath,
    AbductionNetwork,
    ActionNetwork,
    OvRClassifier
)

from .inference import (
    CausalInferenceEngine,
    InferenceValidator
)

from .training import (
    CausalTrainer,
    LossComputer,
    TrainingValidator
)

__version__ = "0.1.0"
__author__ = "CausalQwen Development Team"

# 主要导出
__all__ = [
    # 核心模型
    'CausalQwenMVPForCausalLM',
    'CausalQwen2Config', 
    'CausalMVPOutput',
    
    # 数学工具
    'CauchyMath',
    
    # 网络模块
    'AbductionNetwork',
    'ActionNetwork',
    'OvRClassifier',
    
    # 推理引擎
    'CausalInferenceEngine',
    'InferenceValidator',
    
    # 训练工具
    'CausalTrainer',
    'LossComputer', 
    'TrainingValidator',
]

def create_mvp_model(model_name_or_path="Qwen/Qwen2.5-0.5B", **config_kwargs):
    """快速创建CausalQwen MVP模型
    
    Args:
        model_name_or_path: 预训练模型路径
        **config_kwargs: 额外的配置参数
    
    Returns:
        CausalQwenMVPForCausalLM: 初始化的模型
    """
    # TODO: 实现从预训练模型加载和权重复制
    config = CausalQwen2Config.from_pretrained(model_name_or_path)
    
    # 设置因果模型参数
    for key, value in config_kwargs.items():
        setattr(config, key, value)
    
    model = CausalQwenMVPForCausalLM(config)
    
    return model

def get_model_info():
    """获取模型信息"""
    info = {
        'name': 'CausalQwen MVP',
        'version': __version__,
        'description': '因果语言模型最小可行产品',
        'features': [
            '继承Qwen2ForCausalLM',
            '三种推理模式',
            '完整训练流程',
            '模块化设计'
        ],
        'status': 'MVP - 使用占位式逻辑',
        'next_steps': [
            '完善Cauchy分布数学',
            '优化权重初始化',
            '添加更多验证指标',
            '性能优化'
        ]
    }
    return info 