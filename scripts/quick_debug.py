#!/usr/bin/env python
"""
快速诊断脚本 - 检查基本功能
"""

import os
import sys
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    
    try:
        from src.models.causal_lm import CausalLMConfig, CausalLanguageModel
        print("  ✅ 导入 CausalLanguageModel")
    except Exception as e:
        print(f"  ❌ 导入 CausalLanguageModel 失败: {e}")
        return False
    
    try:
        from src.data.tokenizer import QwenTokenizerWrapper
        print("  ✅ 导入 QwenTokenizerWrapper")
    except Exception as e:
        print(f"  ❌ 导入 QwenTokenizerWrapper 失败: {e}")
        return False
    
    try:
        from src.utils.distributions import cauchy_nll_loss
        print("  ✅ 导入 cauchy_nll_loss")
    except Exception as e:
        print(f"  ❌ 导入 cauchy_nll_loss 失败: {e}")
        return False
    
    return True

def test_qwen_path():
    """测试Qwen模型路径"""
    print("\n🔍 检查Qwen模型路径...")
    
    qwen_model_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    print(f"  路径: {qwen_model_path}")
    
    if not os.path.exists(qwen_model_path):
        print(f"  ❌ 路径不存在")
        return False
    
    # 检查关键文件
    config_file = os.path.join(qwen_model_path, 'config.json')
    if os.path.exists(config_file):
        print(f"  ✅ config.json 存在")
    else:
        print(f"  ❌ config.json 不存在")
        return False
    
    return True

def test_basic_math():
    """测试基本数学函数"""
    print("\n🔍 测试基本数学函数...")
    
    try:
        # 简单的柯西NLL测试
        from src.utils.distributions import cauchy_nll_loss
        
        target = torch.tensor(3.5)
        loc = torch.tensor(2.0)
        scale = torch.tensor(1.5)
        
        loss = cauchy_nll_loss(target, loc, scale)
        print(f"  ✅ 柯西NLL计算成功: {loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"  ❌ 数学函数测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 CausalQwen 快速诊断")
    print("=" * 50)
    
    # 测试步骤
    tests = [
        ("导入测试", test_imports),
        ("路径检查", test_qwen_path), 
        ("数学函数", test_basic_math)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ {name} 出现异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n📊 诊断结果:")
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有基本测试通过！可以运行完整脚本")
    else:
        print("\n⚠️  存在问题，请先修复后再运行完整脚本")
    
    return all_passed

if __name__ == '__main__':
    main()
