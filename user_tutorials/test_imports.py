"""
导入测试脚本
============

这个脚本测试所有模块是否能正确导入
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试所有关键模块的导入"""
    
    print("🧪 测试模块导入...")
    print("=" * 30)
    
    try:
        print("1. 测试 utils.simple_models...")
        from utils.simple_models import SimpleCausalClassifier, SimpleCausalRegressor, compare_with_sklearn
        print("   ✅ simple_models 导入成功")
    except Exception as e:
        print(f"   ❌ simple_models 导入失败: {e}")
        return False
    
    try:
        print("2. 测试 utils.data_helpers...")
        from utils.data_helpers import generate_classification_data, generate_regression_data, explore_data
        print("   ✅ data_helpers 导入成功")
    except Exception as e:
        print(f"   ❌ data_helpers 导入失败: {e}")
        return False
    
    try:
        print("3. 测试基本功能...")
        # 测试生成数据
        X, y, info = generate_classification_data(n_samples=100, n_features=5, n_classes=3)
        print(f"   ✅ 生成分类数据成功: {X.shape}, {len(y)} 标签")
        
        # 测试模型创建
        model = SimpleCausalClassifier()
        print("   ✅ 创建分类器成功")
        
        # 测试回归
        regressor = SimpleCausalRegressor()
        print("   ✅ 创建回归器成功")
        
    except Exception as e:
        print(f"   ❌ 基本功能测试失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！模块导入正常。")
    return True

if __name__ == "__main__":
    success = test_imports()
    
    if success:
        print("\n✅ 您现在可以运行用户教程了！")
        print("\n推荐运行顺序:")
        print("1. python user_tutorials/01_quick_start/installation.py")
        print("2. python user_tutorials/01_quick_start/first_example.py")
        print("3. python user_tutorials/run_user_tutorials.py")
    else:
        print("\n❌ 导入测试失败，请检查文件结构！")
        sys.exit(1)