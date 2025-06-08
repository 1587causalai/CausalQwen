#!/usr/bin/env python3
"""
简单评估测试 - 验证简化后的指标输出
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.synthetic import SyntheticMathDataset
from models.causal_lm import CausalLanguageModel
from models.config import ModelConfig
from evaluation.evaluator import Evaluator
from data.tokenizer import MockTokenizer


def test_simplified_metrics():
    """测试简化后的指标输出"""
    print("=== 测试简化后的指标输出 ===")
    
    # 使用小型配置
    tokenizer = MockTokenizer(vocab_size=100)
    config = ModelConfig(
        causal_dim=16,
        vocab_size=tokenizer.vocab_size,
        num_token_id=tokenizer.num_token_id,
        context_length=8
    )
    
    # 创建模型和评估器
    model = CausalLanguageModel(config)
    evaluator = Evaluator(model, tokenizer, "cpu")
    
    # 生成小数据集
    dataset = SyntheticMathDataset(num_samples=20, context_length=8)
    
    # 评估
    print("运行评估...")
    results = evaluator.evaluate(dataset, batch_size=8)
    
    # 显示结果
    print(f"\n📊 简化后的指标:")
    print(f"分类指标:")
    print(f"  - cls_accuracy: {results['cls_accuracy']:.4f}")
    print(f"  - cls_f1: {results['cls_f1']:.4f}")
    
    print(f"回归指标:")
    print(f"  - reg_mse: {results['reg_mse']:.4f}")
    print(f"  - reg_mae: {results['reg_mae']:.4f}")
    
    print(f"校准指标:")
    print(f"  - calib_ece: {results['calib_ece']:.4f} (分类校准)")
    print(f"  - reg_picp: {results['reg_picp']:.4f} (回归校准)")
    
    # 检查是否还有多余的指标
    unexpected_metrics = []
    for key in results.keys():
        if key.startswith('calib_') and key not in ['calib_ece']:
            unexpected_metrics.append(key)
    
    if unexpected_metrics:
        print(f"\n⚠️  发现多余的校准指标: {unexpected_metrics}")
    else:
        print(f"\n✅ 校准指标已正确简化！")
    
    print(f"\n总计指标数量: {len(results)}")
    print(f"所有指标: {list(results.keys())}")
    
    return results


if __name__ == "__main__":
    test_simplified_metrics() 