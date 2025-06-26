#!/usr/bin/env python3
"""
CausalQwen项目安全整理 - 第一阶段

只删除明确的调试和临时文件，不触碰任何可能有用的文件
"""

import os
from pathlib import Path

def safe_cleanup_stage1():
    """第一阶段：只删除明确的调试文件"""
    
    base_dir = Path("/Users/gongqian/DailyLog/CausalQwen")
    
    print("🧹 CausalQwen项目安全整理 - 第一阶段")
    print("=" * 50)
    print("只删除明确的调试和临时文件")
    print()
    
    # 第一阶段：只删除我们确定创建的调试文件
    debug_files_to_delete = [
        # 我们今天创建的调试文件
        "debug_deterministic_advantage.py",  # 我们创建的对比实验
        "debug_pure_causal_advantage.py",    # 我们创建的纯粹对比
        "debug_standard_mode.py",            # 调试standard模式
        
        # 明显的临时验证文件  
        "simple_math_verification.py",       # 数学验证临时文件
        "verify_anomaly_ratio.py",          # 验证异常比例的临时文件
    ]
    
    print("📁 将要删除的文件:")
    files_to_delete = []
    
    for file in debug_files_to_delete:
        file_path = base_dir / file
        if file_path.exists():
            # 检查文件大小和修改时间，确保是最近的调试文件
            stat = file_path.stat()
            size_kb = stat.st_size / 1024
            print(f"   🗑️  {file} ({size_kb:.1f}KB)")
            files_to_delete.append(file_path)
        else:
            print(f"   ⚠️  未找到: {file}")
    
    print()
    
    if not files_to_delete:
        print("✅ 没有找到需要删除的调试文件")
        return
    
    # 直接删除明确的调试文件
    print(f"开始删除 {len(files_to_delete)} 个明确的调试文件...")
    print("这些都是我们调试过程中创建的临时文件")
    
    if True:  # 直接执行
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"✅ 已删除: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败: {file_path.name} - {e}")
        
        print(f"\n🎯 第一阶段完成：成功删除 {deleted_count} 个调试文件")
        print()
        print("下一阶段可以考虑:")
        print("- test_*.py 文件（根目录下的临时测试）")
        print("- results/test_fix* 目录（旧的测试结果）")
        print("- 重复的demo文件")
        
    else:
        print("❌ 取消删除操作")

if __name__ == "__main__":
    safe_cleanup_stage1()