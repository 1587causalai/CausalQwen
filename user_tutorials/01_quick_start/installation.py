"""
环境安装和验证
=============

这个脚本帮助您验证 CausalQwen 的运行环境是否正确配置。

运行前请确保已安装以下依赖：
pip install torch scikit-learn matplotlib pandas numpy seaborn
"""

import sys
import importlib

def check_package(package_name, import_name=None, min_version=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        if hasattr(module, '__version__'):
            version = module.__version__
            print(f"✅ {package_name}: {version}")
            
            if min_version and version < min_version:
                print(f"   ⚠️  建议版本 >= {min_version}")
        else:
            print(f"✅ {package_name}: 已安装")
        
        return True
    except ImportError:
        print(f"❌ {package_name}: 未安装")
        return False

def main():
    """检查所有必需的包"""
    
    print("🔍 CausalQwen 环境检查")
    print("=" * 40)
    
    # 必需的包
    required_packages = [
        ('Python', None, '3.7'),
        ('torch', 'torch', '1.8.0'),
        ('numpy', 'numpy', '1.19.0'),
        ('pandas', 'pandas', '1.2.0'),
        ('scikit-learn', 'sklearn', '0.24.0'),
        ('matplotlib', 'matplotlib', '3.3.0'),
    ]
    
    # 可选的包（用于更好的体验）
    optional_packages = [
        ('seaborn', 'seaborn'),
        ('jupyter', 'jupyter'),
        ('ipython', 'IPython'),
    ]
    
    print("\\n📦 必需包检查:")
    all_required_ok = True
    for package_name, import_name, min_version in required_packages:
        if package_name == 'Python':
            version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            print(f"✅ Python: {version}")
            if sys.version_info < (3, 7):
                print(f"   ⚠️  建议版本 >= 3.7")
        else:
            ok = check_package(package_name, import_name, min_version)
            if not ok:
                all_required_ok = False
    
    print("\\n📦 可选包检查:")
    for package_name, import_name in optional_packages:
        check_package(package_name, import_name)
    
    # 基本功能测试
    print("\\n🧪 基本功能测试:")
    
    if all_required_ok:
        try:
            # 测试 numpy
            import numpy as np
            arr = np.random.randn(100)
            print("✅ NumPy: 数组运算正常")
            
            # 测试 torch
            import torch
            tensor = torch.randn(10, 10)
            result = torch.matmul(tensor, tensor.T)
            print("✅ PyTorch: 张量运算正常")
            
            # 测试 sklearn
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            print("✅ scikit-learn: 数据生成正常")
            
            # 测试 matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plt.close(fig)  # 立即关闭以避免显示
            print("✅ Matplotlib: 绘图功能正常")
            
            # 测试 pandas
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            print("✅ Pandas: 数据处理正常")
            
        except Exception as e:
            print(f"❌ 功能测试失败: {e}")
            all_required_ok = False
    
    # 简单性能测试
    if all_required_ok:
        print("\\n⚡ 性能测试:")
        
        try:
            import time
            import torch
            
            # CPU 测试
            start_time = time.time()
            x = torch.randn(1000, 1000)
            y = torch.matmul(x, x.T)
            cpu_time = time.time() - start_time
            print(f"✅ CPU 计算: {cpu_time:.3f} 秒")
            
            # GPU 测试（如果可用）
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
                x_gpu = x.cuda()
                start_time = time.time()
                y_gpu = torch.matmul(x_gpu, x_gpu.T)
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                print(f"✅ GPU 计算: {gpu_time:.3f} 秒 ({device})")
                print(f"   GPU 加速比: {cpu_time/gpu_time:.1f}x")
            else:
                print("ℹ️  GPU: 未检测到 CUDA 设备")
        
        except Exception as e:
            print(f"⚠️  性能测试遇到问题: {e}")
    
    # 总结
    print("\\n" + "=" * 40)
    if all_required_ok:
        print("🎉 环境检查完成！您可以开始使用 CausalQwen 了。")
        print("\\n📖 下一步:")
        print("   运行: python user_tutorials/01_quick_start/first_example.py")
    else:
        print("❌ 环境配置不完整！")
        print("\\n🔧 解决方案:")
        print("   1. 安装缺失的包:")
        print("      pip install torch scikit-learn matplotlib pandas numpy")
        print("   2. 或使用 conda:")
        print("      conda install pytorch scikit-learn matplotlib pandas numpy")
        print("   3. 重新运行此脚本验证")

if __name__ == "__main__":
    main()