"""
CausalQwen: 因果语言模型 - 完全兼容Qwen

核心机制：位置vs尺度的差异化处理
├─ do_sample=True：噪声影响位置参数，扰动个体身份
└─ do_sample=False：噪声影响尺度参数，增加决策不确定性

与Qwen完全兼容：
- 继承Qwen2ForCausalLM，无缝接入
- 支持所有Qwen参数：do_sample, temperature, top_k, top_p
- 提供相同的generate()接口
- 完整的柯西分布数学基础

使用示例：
```python
from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config

# 创建模型（与Qwen相同）
config = CausalQwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B")
model = CausalQwenMVPForCausalLM(config)

# 确定性生成（噪声影响尺度参数）
output = model.generate(input_ids, do_sample=False)

# 采样生成（噪声影响位置参数）
output = model.generate(input_ids, do_sample=True, temperature=0.8)
```
"""

# 配置和数据结构
from .config import (
    CausalQwen2Config,
    CausalMVPOutput
)

# 核心组件
from .components import (
    CauchyMath,
    AbductionNetwork,
    ActionNetwork,
    OvRClassifier
)

# 主模型
from .models import (
    CausalQwenMVPForCausalLM
)

# 推理和训练工具
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
            '位置vs尺度的差异化处理',
            'ActionNetwork统一框架',
            '完整柯西分布数学基础',
            '与Qwen完全兼容'
        ],
        'status': '正式版 - 与Qwen完全兼容',
        'next_steps': [
            '完善Cauchy分布数学',
            '优化权重初始化',
            '添加更多验证指标',
            '性能优化'
        ]
    }
    return info 