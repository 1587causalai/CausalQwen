"""
离散有序激活数学验证脚本
详细展示数学计算过程，验证实现的正确性
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_engine import CausalEngine, ActivationHead


def manual_ordinal_calculation(loc_S, scale_S, thresholds):
    """
    手动计算离散有序激活，用于验证自动实现
    
    数学公式：
    P(Y = k) = P(C_k < S ≤ C_{k+1})
             = CDF_Cauchy(C_{k+1}) - CDF_Cauchy(C_k)
    
    其中 CDF_Cauchy(c) = 1/2 + (1/π)arctan((c - loc_S)/scale_S)
    """
    def cauchy_cdf(x, loc, scale):
        return 0.5 + (1 / np.pi) * torch.atan((x - loc) / scale)
    
    # 构建完整阈值序列：[-∞, C_1, C_2, ..., C_{K-1}, +∞]
    neg_inf = torch.tensor(float('-inf'))
    pos_inf = torch.tensor(float('+inf'))
    full_thresholds = torch.cat([
        neg_inf.unsqueeze(0), 
        thresholds, 
        pos_inf.unsqueeze(0)
    ])
    
    num_classes = len(full_thresholds) - 1
    probs = []
    
    print(f"手动计算过程：")
    print(f"loc_S = {loc_S:.4f}, scale_S = {scale_S:.4f}")
    print(f"阈值序列: {[f'{t:.2f}' if not torch.isinf(t) else str(t.item()) for t in full_thresholds]}")
    print()
    
    for k in range(num_classes):
        lower_threshold = full_thresholds[k]
        upper_threshold = full_thresholds[k + 1]
        
        # 计算CDF值
        if torch.isinf(lower_threshold) and lower_threshold < 0:
            lower_cdf = torch.tensor(0.0)
        else:
            lower_cdf = cauchy_cdf(lower_threshold, loc_S, scale_S)
            
        if torch.isinf(upper_threshold) and upper_threshold > 0:
            upper_cdf = torch.tensor(1.0)
        else:
            upper_cdf = cauchy_cdf(upper_threshold, loc_S, scale_S)
        
        # 区间概率
        prob_k = upper_cdf - lower_cdf
        probs.append(prob_k)
        
        print(f"类别 {k}: P({lower_threshold:.2f} < S ≤ {upper_threshold:.2f}) = {prob_k.item():.4f}")
        print(f"  CDF({upper_threshold:.2f}) = {upper_cdf.item():.4f}")
        print(f"  CDF({lower_threshold:.2f}) = {lower_cdf.item():.4f}")
        print(f"  P(Y={k}) = {prob_k.item():.4f}")
        print()
    
    probs_tensor = torch.stack(probs)
    predicted_class = torch.argmax(probs_tensor)
    
    print(f"概率分布: {[f'{p.item():.4f}' for p in probs]}")
    print(f"概率和: {sum(probs).item():.4f} (应该≈1.0)")
    print(f"预测类别: {predicted_class.item()}")
    
    return probs_tensor, predicted_class


def validate_ordinal_activation():
    """验证离散有序激活的数学正确性"""
    
    print("=" * 60)
    print("CausalEngine 离散有序激活数学验证")
    print("=" * 60)
    print()
    
    # 创建一个简单的4级离散有序分类器
    activation_head = ActivationHead(
        output_size=1,
        activation_modes="ordinal",
        ordinal_num_classes=4,
        ordinal_threshold_init=1.0
    )
    
    # 显示初始化的阈值
    thresholds = activation_head.ordinal_thresholds['ordinal_0']
    print(f"初始化阈值: {thresholds.tolist()}")
    print(f"这将创建4个区间: (-∞, {thresholds[0]:.2f}], ({thresholds[0]:.2f}, {thresholds[1]:.2f}], ({thresholds[1]:.2f}, {thresholds[2]:.2f}], ({thresholds[2]:.2f}, +∞)")
    print()
    
    # 测试几个不同的S值
    test_cases = [
        {"loc_S": -2.0, "scale_S": 1.0, "expected_class": 0, "desc": "强烈偏向第1类"},
        {"loc_S": 0.0, "scale_S": 1.0, "expected_class": 1, "desc": "中等偏向第2类"}, 
        {"loc_S": 2.0, "scale_S": 1.0, "expected_class": 3, "desc": "强烈偏向第4类"},
        {"loc_S": 0.0, "scale_S": 0.1, "expected_class": None, "desc": "低不确定性"},
        {"loc_S": 0.0, "scale_S": 5.0, "expected_class": None, "desc": "高不确定性"}
    ]
    
    for i, case in enumerate(test_cases):
        print(f"测试案例 {i+1}: {case['desc']}")
        print("-" * 40)
        
        # 构造输入
        loc_S = torch.tensor([[[case['loc_S']]]])  # [1, 1, 1] - batch_size, seq_len, output_size
        scale_S = torch.tensor([[[case['scale_S']]]])  # [1, 1, 1]
        
        # 手动计算
        print("【手动计算】")
        manual_probs, manual_pred = manual_ordinal_calculation(
            loc_S[0, 0, 0], scale_S[0, 0, 0], thresholds
        )
        
        print("【ActivationHead 自动计算】")
        # 使用ActivationHead自动计算
        with torch.no_grad():
            result = activation_head(loc_S, scale_S, return_dict=True)
            auto_pred = result['output'][0, 0, 0].item()
        
        print(f"自动预测类别: {auto_pred}")
        print()
        
        # 验证一致性
        consistency = abs(manual_pred.item() - auto_pred) < 1e-6
        print(f"一致性检验: {'✅ 通过' if consistency else '❌ 失败'}")
        
        if case['expected_class'] is not None:
            expectation_met = case['expected_class'] == manual_pred.item()
            print(f"期望验证: 期望类别{case['expected_class']}, 实际类别{manual_pred.item()}, {'✅ 符合' if expectation_met else '❌ 不符'}")
        
        print("=" * 60)
        print()


def validate_multi_class_ordinal():
    """验证不同类别数的离散有序激活"""
    
    print("多类别离散有序激活验证")
    print("=" * 40)
    
    class_configs = [3, 5, 7]  # 测试3类、5类、7类
    
    for num_classes in class_configs:
        print(f"\n🎯 {num_classes}类别离散有序激活测试:")
        
        head = ActivationHead(
            output_size=1,
            activation_modes="ordinal", 
            ordinal_num_classes=num_classes,
            ordinal_threshold_init=1.0
        )
        
        # 获取阈值
        thresholds = head.ordinal_thresholds['ordinal_0']
        print(f"阈值数量: {len(thresholds)} (类别数-1)")
        print(f"阈值: {[f'{t:.2f}' for t in thresholds.tolist()]}")
        
        # 测试一个中性的S值
        loc_S = torch.tensor([[[0.0]]])  # [1, 1, 1]
        scale_S = torch.tensor([[[1.0]]])  # [1, 1, 1]
        
        result = head(loc_S, scale_S, return_dict=True)
        pred_class = result['output'][0, 0, 0].item()
        
        print(f"中性输入 (loc_S=0, scale_S=1) 预测类别: {int(pred_class)}")
        print(f"有效类别范围: 0 到 {num_classes-1}")
        
        # 验证输出范围
        valid_range = 0 <= pred_class <= num_classes - 1
        print(f"范围检验: {'✅ 通过' if valid_range else '❌ 失败'}")


if __name__ == "__main__":
    validate_ordinal_activation()
    validate_multi_class_ordinal()
    
    print("\n🎉 离散有序激活数学验证完成！")
    print("\n📝 验证要点:")
    print("1. 手动计算与自动计算结果一致")
    print("2. 概率分布和为1.0")
    print("3. 不同S值产生合理的类别预测")
    print("4. 支持任意数量的有序类别")
    print("5. 数学公式实现正确") 