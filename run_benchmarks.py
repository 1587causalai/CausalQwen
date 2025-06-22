#!/usr/bin/env python3
"""
CausalEngine 基准测试运行脚本
=============================

方便从项目根目录运行基准测试的包装脚本。
"""

import os
import sys
import subprocess

def run_script(script_path, script_name):
    """运行指定脚本"""
    print(f"🚀 运行 {script_name}...")
    print("=" * 60)
    
    try:
        # 更改到 user_tutorials 目录
        os.chdir("user_tutorials")
        
        # 运行脚本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {script_name} 运行完成")
        else:
            print(f"❌ {script_name} 运行失败 (返回码: {result.returncode})")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行 {script_name} 时出错: {e}")
        return False
    finally:
        # 回到原目录
        os.chdir("..")

def main():
    """主函数"""
    print("🔬 CausalEngine 基准测试")
    print("========================")
    
    # 检查当前目录
    if not os.path.exists("user_tutorials"):
        print("❌ 请在 CausalQwen 项目根目录下运行此脚本")
        return
    
    scripts = [
        ("04_real_world_examples/classification_benchmark.py", "分类任务基准测试"),
        ("04_real_world_examples/regression_benchmark.py", "回归任务基准测试")
    ]
    
    print(f"将运行 {len(scripts)} 个基准测试脚本:\n")
    
    for i, (script_path, script_name) in enumerate(scripts, 1):
        print(f"{i}. {script_name}: {script_path}")
    
    print("\n" + "=" * 60)
    
    success_count = 0
    
    for script_path, script_name in scripts:
        if run_script(script_path, script_name):
            success_count += 1
        print()
    
    print("🎉 基准测试完成!")
    print(f"✅ 成功: {success_count}/{len(scripts)} 个脚本")
    
    if success_count == len(scripts):
        print("\n📊 查看结果:")
        print("   - 图表: user_tutorials/results/*.png")
        print("   - 数据: user_tutorials/results/*.csv")
    else:
        print(f"\n⚠️ 有 {len(scripts) - success_count} 个脚本运行失败")

if __name__ == "__main__":
    main()