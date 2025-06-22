"""
用户教程验证脚本
===============

这个脚本验证所有用户教程是否能正常运行
"""

import os
import sys
import subprocess
import time

def run_test(script_path, args=None, timeout=120):
    """运行测试脚本"""
    try:
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        print(f"  🧪 运行: {script_path}")
        start_time = time.time()
        
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=timeout,
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"     ✅ 成功 ({elapsed_time:.1f}s)")
            return True
        else:
            print(f"     ❌ 失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"     错误: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"     ⏰ 超时 (>{timeout}s)")
        return False
    except Exception as e:
        print(f"     ❌ 异常: {e}")
        return False

def main():
    """运行所有验证测试"""
    
    print("🔬 CausalQwen 用户教程验证")
    print("=" * 50)
    
    tests = [
        # 基础测试
        ("test_imports.py", None, "模块导入测试"),
        ("01_quick_start/installation.py", None, "环境检查"),
        ("01_quick_start/test_basic_demo.py", None, "基础功能测试"),
        
        # 主要教程测试（非交互模式）
        ("01_quick_start/first_example.py", ["1"], "第一个示例-分类"),
        ("01_quick_start/first_example.py", ["2"], "第一个示例-回归"),
    ]
    
    # 可选的高级测试（如果时间允许）
    advanced_tests = [
        ("02_classification/synthetic_data.py", ["1"], "合成数据分类"),
        ("03_regression/synthetic_data.py", ["1"], "合成数据回归"),
    ]
    
    print("\\n📋 基础功能测试:")
    basic_results = []
    
    for script, args, description in tests:
        print(f"\\n{description}:")
        # 给示例更多时间
        timeout = 120 if "示例" in description else 60
        success = run_test(script, args, timeout=timeout)
        basic_results.append((description, success))
    
    # 统计基础测试结果
    basic_passed = sum(1 for _, success in basic_results if success)
    basic_total = len(basic_results)
    
    print(f"\\n📊 基础测试结果: {basic_passed}/{basic_total} 通过")
    
    # 如果基础测试都通过，运行高级测试
    if basic_passed == basic_total:
        print("\\n🚀 基础测试全部通过，运行高级测试...")
        
        advanced_results = []
        for script, args, description in advanced_tests:
            print(f"\\n{description}:")
            # 给高级测试更多时间
            success = run_test(script, args, timeout=180)
            advanced_results.append((description, success))
        
        advanced_passed = sum(1 for _, success in advanced_results if success)
        advanced_total = len(advanced_results)
        
        print(f"\\n📊 高级测试结果: {advanced_passed}/{advanced_total} 通过")
        
        total_passed = basic_passed + advanced_passed
        total_tests = basic_total + advanced_total
    else:
        print("\\n⚠️ 基础测试未全部通过，跳过高级测试")
        total_passed = basic_passed
        total_tests = basic_total
    
    print("\\n" + "=" * 50)
    
    if total_passed == total_tests:
        print("🎉 所有测试通过！用户教程可以正常使用。")
        print("\\n✅ 用户可以：")
        print("   1. 运行 python user_tutorials/run_user_tutorials.py")
        print("   2. 直接运行任何教程脚本")
        print("   3. 在自己的项目中使用 utils/ 中的工具")
        return True
    else:
        print(f"❌ {total_tests - total_passed} 个测试失败")
        print("\\n🔧 请检查：")
        print("   1. 所有依赖是否正确安装")
        print("   2. 文件权限是否正确")
        print("   3. Python 版本是否兼容")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)