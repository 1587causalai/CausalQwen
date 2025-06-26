#!/usr/bin/env python3
"""
CausalQwen项目整理脚本

清理历史遗留文件，保持项目结构清晰
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """整理项目，移除历史遗留文件"""
    
    base_dir = Path("/Users/gongqian/DailyLog/CausalQwen")
    
    print("🧹 开始整理CausalQwen项目...")
    print("=" * 60)
    
    # 1. 可以删除的调试和临时文件
    debug_files = [
        "debug_deterministic_advantage.py",
        "debug_pure_causal_advantage.py", 
        "debug_standard_mode.py",
        "simple_math_verification.py",
        "analyze_classification_data.py",
        "demo_causal_applications.py",
        "demo_causal_engine_evaluation.py", 
        "demo_scientific_equivalence_validation.py",
        "demo_sklearn_interface.py",
        "demo_sklearn_interface_v2.py",
        "test_b_noise_trainable.py",
        "test_classifier_init.py", 
        "test_default_init.py",
        "test_scenarios.py",
        "final_b_noise_test.py",
        "verify_anomaly_ratio.py"
    ]
    
    print("📁 删除调试和临时文件:")
    for file in debug_files:
        file_path = base_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"   ✅ 删除: {file}")
        else:
            print(f"   ⚠️  未找到: {file}")
    
    # 2. 清理重复的测试结果
    results_dir = base_dir / "results"
    if results_dir.exists():
        print(f"\n📊 清理测试结果目录...")
        # 只保留最新的结果，删除旧的测试目录
        old_test_dirs = ["test_fix", "test_fix2", "test_fix3", "test_fix4"]
        for test_dir in old_test_dirs:
            test_path = results_dir / test_dir
            if test_path.exists():
                shutil.rmtree(test_path)
                print(f"   ✅ 删除旧测试结果: {test_dir}")
    
    # 3. 整理文档重复
    print(f"\n📚 检查文档重复...")
    docs_dir = base_dir / "docs"
    if docs_dir.exists():
        duplicate_docs = [
            "sklearn_style_api_classifier_v1.md",
            "sklearn_style_api_regressor_v1.md"
        ]
        for doc in duplicate_docs:
            doc_path = docs_dir / doc
            if doc_path.exists():
                doc_path.unlink()
                print(f"   ✅ 删除重复文档: {doc}")
    
    # 4. 修复嵌套目录问题
    user_tutorials_nested = base_dir / "user_tutorials" / "user_tutorials"
    if user_tutorials_nested.exists():
        print(f"\n📁 修复嵌套目录...")
        # 将嵌套的内容移动到正确位置
        target_dir = base_dir / "user_tutorials"
        for item in user_tutorials_nested.iterdir():
            if not (target_dir / item.name).exists():
                shutil.move(str(item), str(target_dir / item.name))
                print(f"   ✅ 移动: {item.name}")
        
        # 删除空的嵌套目录
        if user_tutorials_nested.exists() and not list(user_tutorials_nested.iterdir()):
            user_tutorials_nested.rmdir()
            print(f"   ✅ 删除空目录: user_tutorials/user_tutorials")
    
    # 5. 显示保留的核心结构
    print(f"\n✅ 整理完成！保留的核心结构:")
    print("📦 CausalQwen/")
    print("├── 🧠 causal_engine/          # 核心算法")
    print("├── 🧪 quick_test_causal_engine.py  # 主要测试脚本") 
    print("├── 👥 user_tutorials/         # 用户教程")
    print("├── 🔬 tutorials/             # 开发者教程")
    print("├── 📊 results/               # 测试结果（已清理）")
    print("├── 📚 docs/                  # 文档（已去重）")
    print("├── 🏗️  src/                   # CausalQwen应用层")
    print("├── ⚙️  tests/                # 单元测试")
    print("└── 📋 README.md              # 项目说明")
    
    print(f"\n🎯 项目现在更加清晰，专注于核心功能！")
    print("下一步建议:")
    print("1. 运行 quick_test_causal_engine.py 验证核心功能")
    print("2. 检查 user_tutorials/ 确保用户体验完整")
    print("3. 更新 README.md 反映当前项目状态")

if __name__ == "__main__":
    # 安全检查
    response = input("确定要整理项目吗？这将删除一些调试文件 (y/n): ")
    if response.lower() == 'y':
        cleanup_project()
    else:
        print("取消整理操作")