#!/usr/bin/env python3
"""
CausalEngine 教程快速启动脚本

这个脚本提供了一个交互式菜单，让用户可以轻松选择和运行不同的教程。
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    print("="*70)
    print("🎯 CausalEngine 实际应用教程")
    print("   展示因果推理在机器学习中的强大威力")
    print("="*70)


def print_menu():
    """打印教程菜单"""
    tutorials = [
        {
            "id": "1",
            "name": "表格数据快速测试",
            "path": "01_classification/tabular_quick_test.py",
            "description": "在5个经典小数据集上快速对比CausalEngine性能",
            "difficulty": "🟢 基础",
            "time": "~2分钟"
        },
        {
            "id": "1b",
            "name": "表格数据完整基准测试",
            "path": "01_classification/tabular_classification_benchmark.py",
            "description": "在4个大型数据集上全面测试，已修复macOS兼容性",
            "difficulty": "🔥 进阶",
            "time": "~8分钟"
        },
        {
            "id": "2", 
            "name": "房价预测 (回归任务)",
            "path": "02_regression/house_price_prediction.py",
            "description": "California Housing回归，展示柯西分布鲁棒性和预测区间",
            "difficulty": "🟢 基础",
            "time": "~4分钟"
        },
        {
            "id": "3",
            "name": "评分预测 (有序分类)",
            "path": "03_ordinal/rating_prediction.py", 
            "description": "电影星级评分，展示新的离散有序激活功能 (v2.0.4)",
            "difficulty": "🟢 基础",
            "time": "~3分钟"
        },
        {
            "id": "4",
            "name": "电商分析 (多任务学习)",
            "path": "04_multitask/ecommerce_analysis.py",
            "description": "同时预测情感+评分+有用性，展示混合激活模式",
            "difficulty": "🔥 进阶", 
            "time": "~5分钟"
        },
        {
            "id": "0",
            "name": "查看教程概览",
            "path": "README.md",
            "description": "打开教程文档，了解完整教程体系",
            "difficulty": "📖 文档",
            "time": "~2分钟"
        }
    ]
    
    print("\n📚 可用教程:")
    print("-" * 70)
    
    for tutorial in tutorials:
        print(f"[{tutorial['id']}] {tutorial['name']}")
        print(f"    📝 {tutorial['description']}")
        print(f"    🎯 难度: {tutorial['difficulty']} | ⏱️ 预计时间: {tutorial['time']}")
        print()
    
    print("[q] 退出")
    print("-" * 70)
    
    return tutorials


def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'torch',
        'numpy', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_causal_engine():
    """检查 CausalEngine 是否可以正常导入"""
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from causal_engine import CausalEngine
        print("✅ CausalEngine 导入成功")
        return True
    except ImportError as e:
        print(f"❌ CausalEngine 导入失败: {e}")
        print("请确保你在正确的目录中运行此脚本")
        return False


def run_tutorial(tutorial_path):
    """运行指定的教程"""
    tutorial_dir = Path(__file__).parent
    full_path = tutorial_dir / tutorial_path
    
    if not full_path.exists():
        print(f"❌ 教程文件不存在: {full_path}")
        return False
    
    if tutorial_path.endswith('.md'):
        # 打开文档文件
        if sys.platform.startswith('darwin'):  # macOS
            subprocess.run(['open', str(full_path)])
        elif sys.platform.startswith('linux'):  # Linux
            subprocess.run(['xdg-open', str(full_path)])
        elif sys.platform.startswith('win'):   # Windows
            subprocess.run(['start', str(full_path)], shell=True)
        else:
            print(f"📖 请手动打开文档: {full_path}")
        return True
    
    try:
        print(f"🚀 正在运行教程: {tutorial_path}")
        print("=" * 50)
        
        # 切换到教程目录
        old_cwd = os.getcwd()
        os.chdir(full_path.parent)
        
        # 运行教程
        result = subprocess.run([sys.executable, full_path.name], 
                              capture_output=False, text=True)
        
        # 恢复原目录
        os.chdir(old_cwd)
        
        if result.returncode == 0:
            print("\n✅ 教程运行完成!")
            print("📊 请查看生成的可视化图片了解详细结果")
        else:
            print(f"\n❌ 教程运行出错 (退出码: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行教程时出错: {e}")
        return False


def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    print("🔍 检查环境...")
    if not check_dependencies():
        return
    
    if not check_causal_engine():
        return
    
    while True:
        tutorials = print_menu()
        
        try:
            choice = input("请选择教程 (输入数字或字母): ").strip()
            
            if choice.lower() == 'q':
                print("👋 再见！感谢使用 CausalEngine 教程!")
                break
            
            # 查找选中的教程
            selected_tutorial = None
            for tutorial in tutorials:
                if tutorial['id'] == choice:
                    selected_tutorial = tutorial
                    break
            
            if selected_tutorial is None:
                print("❌ 无效选择，请重新输入")
                continue
            
            print(f"\n📌 你选择了: {selected_tutorial['name']}")
            print(f"📝 {selected_tutorial['description']}")
            
            # 运行教程
            success = run_tutorial(selected_tutorial['path'])
            
            if success:
                input("\n按 Enter 键返回主菜单...")
            else:
                input("\n按 Enter 键返回主菜单 (检查错误信息)...")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 出现错误: {e}")
            input("按 Enter 键继续...")


if __name__ == "__main__":
    main()