"""
用户教程运行器
=============

这个脚本提供了一个简单的菜单界面，让用户可以轻松选择和运行不同的教程。

使用方法:
python user_tutorials/run_user_tutorials.py
"""

import os
import sys
import subprocess

def run_script(script_path):
    """运行Python脚本"""
    try:
        print(f"\\n🚀 正在运行: {script_path}")
        print("=" * 50)
        
        # 确保使用绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_script_path = os.path.join(current_dir, script_path)
        
        # 运行脚本
        result = subprocess.run([sys.executable, full_script_path], 
                              capture_output=False, 
                              text=True, 
                              cwd=current_dir)
        
        if result.returncode == 0:
            print("\\n✅ 脚本运行完成!")
        else:
            print(f"\\n❌ 脚本运行出错，返回码: {result.returncode}")
            
    except Exception as e:
        print(f"\\n❌ 运行脚本时出错: {e}")

def main():
    """主菜单"""
    
    print("🌟 CausalEngine 用户教程")
    print("=" * 40)
    
    print("\\n欢迎使用 CausalEngine 因果推理引擎！")
    print("这里有一系列循序渐进的教程，帮助您快速上手。")
    
    while True:
        print("\\n📚 教程菜单:")
        print("\\n🚀 快速开始:")
        print("  1. 环境检查 - 验证安装是否正确")
        print("  2. 第一个示例 - 5分钟体验 CausalQwen")
        
        print("\\n🎯 分类任务:")
        print("  3. 合成数据分类 - 学习分类技巧")
        print("  4. 鸢尾花分类 - 真实数据实战")
        
        print("\\n📈 回归任务:")
        print("  5. 合成数据回归 - 学习回归技巧")
        print("  6. 房价预测 - 真实数据实战")
        
        print("\\n🔧 实用工具:")
        print("  7. 数据处理演示 - 了解工具函数")
        print("  8. 模型对比演示 - 与传统方法对比")
        
        print("\\n其他选项:")
        print("  9. 查看所有教程文件")
        print("  0. 退出")
        
        try:
            choice = input("\\n请选择 (0-9): ").strip()
            
            if choice == '0':
                print("\\n👋 感谢使用 CausalEngine！")
                break
            elif choice == '1':
                run_script("01_quick_start/installation.py")
            elif choice == '2':
                run_script("01_quick_start/first_example.py")
            elif choice == '3':
                run_script("02_classification/synthetic_data.py")
            elif choice == '4':
                run_script("02_classification/iris_dataset.py")
            elif choice == '5':
                run_script("03_regression/synthetic_data.py")
            elif choice == '6':
                run_script("03_regression/boston_housing.py")
            elif choice == '7':
                demo_data_helpers()
            elif choice == '8':
                demo_model_comparison()
            elif choice == '9':
                show_all_tutorials()
            else:
                print("❌ 无效选择，请输入 0-9 之间的数字")
                
        except KeyboardInterrupt:
            print("\\n\\n👋 用户取消，退出程序")
            break
        except Exception as e:
            print(f"\\n❌ 发生错误: {e}")

def demo_data_helpers():
    """演示数据处理工具"""
    print("\\n🔧 数据处理工具演示")
    print("=" * 30)
    
    try:
        # 导入工具
        sys.path.append(os.path.dirname(__file__))
        from utils.data_helpers import generate_classification_data, explore_data
        
        print("\\n1. 生成示例分类数据...")
        X, y, info = generate_classification_data(n_samples=200, n_features=5, n_classes=3)
        
        print("\\n2. 数据探索...")
        explore_data(X, y, info, show_plots=True)
        
        print("\\n✅ 数据处理工具演示完成！")
        print("💡 您可以在自己的代码中使用这些工具函数。")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")

def demo_model_comparison():
    """演示模型对比"""
    print("\\n⚖️ 模型对比演示")
    print("=" * 30)
    
    try:
        # 导入工具
        sys.path.append(os.path.dirname(__file__))
        from utils.simple_models import compare_with_sklearn
        from utils.data_helpers import generate_classification_data
        
        print("\\n1. 生成测试数据...")
        X, y, _ = generate_classification_data(n_samples=500, n_features=10, n_classes=3)
        
        print("\\n2. 运行模型对比...")
        results = compare_with_sklearn(X, y, task_type='classification')
        
        print("\\n✅ 模型对比演示完成！")
        print("💡 CausalQwen 通常在复杂数据上表现更好。")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")

def show_all_tutorials():
    """显示所有教程文件"""
    print("\\n📁 所有教程文件:")
    print("=" * 30)
    
    tutorial_structure = {
        "01_quick_start/": [
            "installation.py - 环境检查工具",
            "first_example.py - 第一个完整示例"
        ],
        "02_classification/": [
            "synthetic_data.py - 合成数据分类教程",
            "iris_dataset.py - 鸢尾花分类实战"
        ],
        "03_regression/": [
            "synthetic_data.py - 合成数据回归教程", 
            "boston_housing.py - 房价预测实战"
        ],
        "utils/": [
            "simple_models.py - 用户友好的模型接口",
            "data_helpers.py - 数据处理工具函数"
        ]
    }
    
    for folder, files in tutorial_structure.items():
        print(f"\\n📂 {folder}")
        for file_info in files:
            print(f"   📄 {file_info}")
    
    print("\\n💡 提示:")
    print("   - 建议按顺序学习，从 01_quick_start 开始")
    print("   - 每个教程都包含详细的注释和说明")
    print("   - utils/ 中的工具可以在您自己的项目中使用")

if __name__ == "__main__":
    # 确保在正确的目录中运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()