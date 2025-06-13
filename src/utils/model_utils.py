#!/usr/bin/env python
"""
模型工具函数 (Model Utilities)

提供用于处理和检查 Hugging Face 模型配置和参数的辅助函数。
"""
from transformers import AutoConfig
from typing import Dict, Optional

def get_qwen_model_info(model_path: str) -> Optional[Dict[str, int]]:
    """
    从给定的模型路径加载 Qwen 模型的配置，并提取关键参数。

    这种方法避免了硬编码模型参数（如 vocab_size, hidden_size），
    使得代码库可以轻松适应不同大小的 Qwen 模型 (e.g., 0.5B, 1.8B, 4B)。

    Args:
        model_path (str): 指向 Hugging Face 模型目录的路径。

    Returns:
        Optional[Dict[str, int]]: 包含 'vocab_size' 和 'hidden_size' 的字典，
                                  如果加载失败则返回 None。
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
        }
    except Exception as e:
        print(f"❌ 无法从 '{model_path}' 加载模型配置: {e}")
        return None 