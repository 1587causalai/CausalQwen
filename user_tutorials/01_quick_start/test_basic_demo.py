"""
基础演示测试 - 非交互版本
=======================

这个脚本自动运行分类和回归演示，无需用户交互，
用于验证教程功能是否正常工作。
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalClassifier, SimpleCausalRegressor
from utils.data_helpers import generate_classification_data, generate_regression_data

def test_classification():
    """测试分类功能"""
    print("🎯 测试分类任务")
    print("-" * 30)
    
    # 生成数据
    X, y, info = generate_classification_data(
        n_samples=200,
        n_features=8,
        n_classes=3,
        difficulty='easy'
    )
    
    # 训练模型
    print("训练分类器...")
    model = SimpleCausalClassifier(random_state=42)
    model.fit(X, y, epochs=20, verbose=False)
    
    # 预测
    predictions = model.predict(X[:10])
    pred_probs = model.predict(X[:10], return_probabilities=True)[1]
    
    print(f"✅ 分类测试完成")
    print(f"   前10个样本预测: {predictions}")
    print(f"   平均置信度: {np.max(pred_probs, axis=1).mean():.3f}")
    
    return True

def test_regression():
    """测试回归功能"""
    print("\n📈 测试回归任务")
    print("-" * 30)
    
    # 生成数据
    X, y, info = generate_regression_data(
        n_samples=200,
        n_features=10,
        noise_level=0.1,
        difficulty='easy'
    )
    
    # 训练模型
    print("训练回归器...")
    model = SimpleCausalRegressor(random_state=42)
    model.fit(X, y, epochs=20, verbose=False)
    
    # 预测
    predictions = model.predict(X[:10])
    
    # 计算简单指标
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y[:10], predictions)
    mae = mean_absolute_error(y[:10], predictions)
    
    print(f"✅ 回归测试完成")
    print(f"   R² 分数: {r2:.4f}")
    print(f"   平均绝对误差: {mae:.4f}")
    
    return True

def main():
    """主测试函数"""
    print("🧪 CausalQwen 基础功能测试")
    print("=" * 40)
    
    try:
        # 测试分类
        classification_ok = test_classification()
        
        # 测试回归
        regression_ok = test_regression()
        
        if classification_ok and regression_ok:
            print("\n🎉 所有测试通过！")
            print("CausalQwen 用户教程功能正常。")
            print("\n📖 您可以运行完整的交互式教程:")
            print("   python user_tutorials/run_user_tutorials.py")
            return True
        else:
            print("\n❌ 部分测试失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)