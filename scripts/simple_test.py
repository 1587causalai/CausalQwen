#!/usr/bin/env python
"""
最简单的测试脚本 - 逐步排查问题
"""

import os
import sys

print("🔍 开始最简单的测试...")

# 1. 基本Python测试
print("1. Python基本功能测试...")
print(f"   Python版本: {sys.version}")
print(f"   当前目录: {os.getcwd()}")

# 2. 路径测试
print("\n2. 路径测试...")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"   项目根目录: {project_root}")
sys.path.insert(0, project_root)

# 3. torch测试
print("\n3. PyTorch测试...")
try:
    import torch
    print(f"   ✅ PyTorch版本: {torch.__version__}")
    print(f"   设备: {torch.device('cpu')}")
except Exception as e:
    print(f"   ❌ PyTorch导入失败: {e}")
    exit(1)

# 4. 基本导入测试
print("\n4. 项目模块导入测试...")
try:
    from src.models.causal_lm import CausalLMConfig
    print("   ✅ CausalLMConfig 导入成功")
except Exception as e:
    print(f"   ❌ CausalLMConfig 导入失败: {e}")
    print(f"   检查路径: {project_root}/src/models/causal_lm.py")
    exit(1)

# 5. Qwen路径测试
print("\n5. Qwen模型路径测试...")
qwen_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
if os.path.exists(qwen_path):
    print(f"   ✅ Qwen路径存在: {qwen_path}")
else:
    print(f"   ❌ Qwen路径不存在: {qwen_path}")
    exit(1)

print("\n🎉 所有基本测试通过！")
print("现在可以运行更复杂的脚本了。")
