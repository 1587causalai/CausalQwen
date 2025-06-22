"""
真实数据集示例 - 加利福尼亚房价预测  
=====================================

这个教程使用加利福尼亚房价数据集，演示如何用 CausalQwen 处理真实的回归任务。
该数据集包含加利福尼亚各地区的房屋和人口信息，目标是预测房屋价值中位数。

学习目标：
1. 学会处理真实的回归数据
2. 理解房价预测的业务背景
3. 掌握特征工程技巧
4. 学会解释房价预测结果
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.simple_models import SimpleCausalRegressor, compare_with_sklearn
from utils.data_helpers import visualize_predictions


def main():
    """主函数：加利福尼亚房价预测完整流程"""
    
    print("🏠 CausalQwen 真实数据示例 - 加利福尼亚房价预测")
    print("=" * 60)
    
    print("\\n📚 关于加利福尼亚房价数据集:")
    print("这是1990年加利福尼亚人口普查的数据，包含20,640个地区的信息。")
    print("每个样本代表一个人口普查区块，包含该区域的房屋和人口统计信息。")
    print("目标是预测该区域房屋价值的中位数（单位：十万美元）。")
    
    # 1. 加载数据
    print("\\n📊 步骤 1: 加载加利福尼亚房价数据集")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    
    feature_names = housing.feature_names
    
    print(f"✅ 数据加载成功:")
    print(f"   样本数量: {X.shape[0]:,}")
    print(f"   特征数量: {X.shape[1]}")
    print(f"   目标变量: 房屋价值中位数（十万美元）")
    
    print(f"\\n🏷️ 特征说明:")
    feature_descriptions = {
        'MedInc': '收入中位数（万美元）',
        'HouseAge': '房屋年龄中位数',
        'AveRooms': '平均房间数',
        'AveBedrms': '平均卧室数',
        'Population': '人口数量',
        'AveOccup': '平均居住人数',
        'Latitude': '纬度',
        'Longitude': '经度'
    }
    
    for i, name in enumerate(feature_names):
        desc = feature_descriptions.get(name, '未知特征')
        print(f"   {i+1}. {name}: {desc}")
    
    # 2. 数据探索
    print("\\n🔍 步骤 2: 数据探索")
    explore_housing_data(X, y, feature_names)
    
    # 3. 特征工程
    print("\\n🔧 步骤 3: 特征工程")
    X_engineered = engineer_features(X, feature_names)
    new_feature_names = feature_names + ['Rooms_per_household', 'Bedrooms_per_room', 'Population_density']
    
    print("\\n新增特征:")
    print("   - Rooms_per_household: 每户房间数 = AveRooms / AveOccup")
    print("   - Bedrooms_per_room: 卧室比例 = AveBedrms / AveRooms") 
    print("   - Population_density: 人口密度的代理变量")
    
    # 4. 数据预处理
    print("\\n🔧 步骤 4: 数据预处理")
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42
    )
    
    # 标准化特征和目标
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"   训练集: {X_train.shape[0]:,} 样本")
    print(f"   测试集: {X_test.shape[0]:,} 样本")
    print(f"   特征和目标均已标准化")
    
    # 5. 训练 CausalQwen 模型
    print("\\n🚀 步骤 5: 训练 CausalQwen 回归器")
    
    model = SimpleCausalRegressor(random_state=42)
    model.fit(X_train_scaled, y_train_scaled, epochs=80, verbose=True)
    
    # 6. 模型评估
    print("\\n📊 步骤 6: 模型评估")
    
    # 预测（标准化空间）
    predictions_scaled = model.predict(X_test_scaled)
    
    # 反标准化到原始空间
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    y_test_original = y_test
    
    # 计算指标
    r2 = r2_score(y_test_original, predictions)
    mae = mean_absolute_error(y_test_original, predictions)
    mse = mean_squared_error(y_test_original, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\\n📈 模型性能指标:")
    print(f"   R² 分数: {r2:.4f}")
    print(f"   平均绝对误差: {mae:.4f} (十万美元)")
    print(f"   均方根误差: {rmse:.4f} (十万美元)")
    print(f"   平均绝对误差: ${mae*100000:.0f}")
    print(f"   均方根误差: ${rmse*100000:.0f}")
    
    # 7. 不确定性分析
    print("\\n🔍 步骤 7: 不确定性分析")
    analyze_uncertainty(model, X_test_scaled, y_test_original, scaler_y)
    
    # 8. 特征重要性分析
    print("\\n📈 步骤 8: 特征重要性分析")
    analyze_feature_importance(model, new_feature_names)
    
    # 9. 预测案例分析
    print("\\n🏠 步骤 9: 预测案例分析")
    analyze_prediction_cases(model, X_test_scaled, y_test_original, new_feature_names, scaler_X, scaler_y)
    
    # 10. 地理分布分析
    print("\\n🗺️ 步骤 10: 地理分布分析")
    analyze_geographic_patterns(X_test, y_test_original, predictions, feature_names)
    
    # 11. 与传统方法对比
    print("\\n⚖️ 步骤 11: 与传统机器学习对比")
    comparison_results = compare_with_sklearn(X_engineered, y, task_type='regression')
    
    # 12. 实际应用演示
    print("\\n🏡 步骤 12: 实际应用演示")
    demo_house_price_prediction(model, scaler_X, scaler_y, new_feature_names)
    
    # 13. 结果可视化
    print("\\n📊 步骤 13: 结果可视化")
    visualize_predictions(y_test_original, predictions, 'regression', '加利福尼亚房价预测 - CausalQwen 结果')
    
    # 绘制详细分析图
    plot_detailed_analysis(y_test_original, predictions, X_test, feature_names)
    
    print("\\n🎉 加利福尼亚房价预测教程完成！")
    print("\\n🎯 关键收获:")
    print(f"   1. 在真实房价数据上达到了 R² = {r2:.4f}")
    print(f"   2. 平均预测误差约为 ${mae*100000:.0f}")
    print("   3. 学会了房价预测的特征工程")
    print("   4. 理解了地理因素对房价的影响")
    print("   5. 掌握了不确定性量化在房价预测中的价值")


def explore_housing_data(X, y, feature_names):
    """探索房价数据"""
    
    # 创建DataFrame便于分析
    df = pd.DataFrame(X, columns=feature_names)
    df['price'] = y
    
    print("\\n基本统计信息:")
    print(df.describe())
    
    print(f"\\n房价分布:")
    print(f"   最低房价: ${y.min()*100000:.0f}")
    print(f"   最高房价: ${y.max()*100000:.0f}")
    print(f"   平均房价: ${y.mean()*100000:.0f}")
    print(f"   房价中位数: ${np.median(y)*100000:.0f}")
    
    # 计算特征与房价的相关性
    print("\\n特征与房价的相关性:")
    correlations = df.corr()['price'].sort_values(key=abs, ascending=False)
    
    for feature, corr in correlations.items():
        if feature != 'price':
            direction = "正相关" if corr > 0 else "负相关"
            strength = "强" if abs(corr) > 0.5 else "中等" if abs(corr) > 0.3 else "弱"
            print(f"   {feature:12}: {corr:6.3f} ({strength}{direction})")


def engineer_features(X, feature_names):
    """特征工程"""
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # 新增特征
    # 1. 每户房间数
    df['Rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    
    # 2. 卧室比例
    df['Bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    
    # 3. 人口密度的代理变量
    df['Population_density'] = df['Population'] / df['AveOccup']
    
    # 处理可能的无穷大或NaN值
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    return df.values


def analyze_uncertainty(model, X_test_scaled, y_test_original, scaler_y):
    """分析预测不确定性"""
    
    # 选择一些样本进行不确定性分析
    sample_indices = np.random.choice(len(X_test_scaled), 20, replace=False)
    X_sample = X_test_scaled[sample_indices]
    y_sample = y_test_original[sample_indices]
    
    # 获取带不确定性的预测
    pred_mean_scaled, pred_std_scaled = model.predict(X_sample, return_uncertainty=True)
    
    # 反标准化
    pred_mean = scaler_y.inverse_transform(pred_mean_scaled.reshape(-1, 1)).flatten()
    pred_std = pred_std_scaled * scaler_y.scale_[0]  # 标准差的缩放
    
    print("\\n房价预测不确定性分析:")
    print("  样本  |  真实价格   |  预测价格   |   不确定性   |   状态")
    print("  ---- | ---------- | ---------- | ----------- | --------")
    
    accurate_count = 0
    for i in range(len(pred_mean)):
        true_price = y_sample[i] * 100000  # 转换为美元
        pred_price = pred_mean[i] * 100000
        uncertainty = pred_std[i] * 100000
        
        # 判断预测是否在不确定性范围内
        error = abs(true_price - pred_price)
        is_accurate = error <= uncertainty
        if is_accurate:
            accurate_count += 1
        
        status = "✅ 准确" if is_accurate else "⚠️ 偏差"
        
        print(f"  {i+1:2d}   | ${true_price:8.0f}  | ${pred_price:8.0f}  | ±${uncertainty:8.0f}  | {status}")
    
    accuracy_rate = accurate_count / len(pred_mean)
    print(f"\\n不确定性校准准确率: {accuracy_rate:.2%}")
    print(f"平均不确定性: ±${np.mean(pred_std)*100000:.0f}")
    
    print("\\n💡 不确定性解释:")
    print("   - 不确定性低的预测更可靠")
    print("   - 不确定性高的区域可能有特殊情况")
    print("   - 在房地产投资决策中可以参考不确定性")


def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    
    if hasattr(model, '_get_feature_importance'):
        importance = model._get_feature_importance()
        if importance is not None:
            print("\\n房价预测特征重要性排序:")
            
            # 按重要性排序
            sorted_indices = np.argsort(importance)[::-1]
            
            for i, idx in enumerate(sorted_indices):
                if i < len(feature_names):
                    print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
            
            # 解释最重要的特征
            if len(sorted_indices) > 0 and sorted_indices[0] < len(feature_names):
                most_important = feature_names[sorted_indices[0]]
                print(f"\\n💡 最重要的特征是 '{most_important}'")
                
                feature_interpretations = {
                    'MedInc': '收入是影响房价的最重要因素',
                    'Latitude': '地理位置（纬度）显著影响房价',
                    'Longitude': '地理位置（经度）显著影响房价',
                    'AveRooms': '房间数量是房价的重要指标',
                    'Rooms_per_household': '房屋空间效率影响价值',
                    'HouseAge': '房屋年龄影响价值评估'
                }
                
                interpretation = feature_interpretations.get(most_important, '这个特征对房价预测很重要')
                print(f"   {interpretation}")
        else:
            print("\\n特征重要性信息暂不可用")
    
    # 基于常识的特征解释
    print("\\n🏠 房价影响因素的常识解释:")
    common_sense = {
        'MedInc': '收入越高的地区，房价通常越高',
        'Latitude/Longitude': '地理位置决定了便利性和环境',
        'AveRooms': '更多房间意味着更大的居住空间',
        'HouseAge': '较新的房屋通常价值更高',
        'Population': '人口密度可能影响房价',
        'AveOccup': '居住密度影响生活质量'
    }
    
    for factor, explanation in common_sense.items():
        print(f"   • {factor}: {explanation}")


def analyze_prediction_cases(model, X_test_scaled, y_test_original, feature_names, scaler_X, scaler_y):
    """分析具体预测案例"""
    
    # 预测所有测试样本
    predictions_scaled = model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    # 计算预测误差
    errors = np.abs(y_test_original - predictions)
    
    # 找到不同类型的案例
    best_idx = np.argmin(errors)  # 最准确的预测
    worst_idx = np.argmax(errors)  # 误差最大的预测
    median_idx = np.argsort(errors)[len(errors)//2]  # 中等误差的预测
    
    cases = [
        (best_idx, "🎯 最准确预测"),
        (median_idx, "📊 典型预测"),
        (worst_idx, "⚠️ 最大误差预测")
    ]
    
    print("\\n代表性预测案例分析:")
    
    for idx, case_name in cases:
        true_price = y_test_original[idx] * 100000
        pred_price = predictions[idx] * 100000
        error = abs(true_price - pred_price)
        error_rate = error / true_price * 100
        
        print(f"\\n{case_name}:")
        print(f"   真实房价: ${true_price:.0f}")
        print(f"   预测房价: ${pred_price:.0f}")
        print(f"   绝对误差: ${error:.0f} ({error_rate:.1f}%)")
        
        # 显示该样本的特征
        original_features = scaler_X.inverse_transform([X_test_scaled[idx]])[0]
        print(f"   区域特征:")
        
        key_features = ['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude']
        for i, name in enumerate(feature_names):
            if i < len(original_features) and (name in key_features or i < 5):
                value = original_features[i]
                print(f"     {name}: {value:.2f}")


def analyze_geographic_patterns(X_test, y_test_original, predictions, feature_names):
    """分析地理分布模式"""
    
    # 找到经纬度的列索引
    lat_idx = feature_names.index('Latitude')
    lon_idx = feature_names.index('Longitude') 
    
    latitudes = X_test[:, lat_idx]
    longitudes = X_test[:, lon_idx]
    
    print("\\n地理分布分析:")
    
    # 按纬度分析（南北方向）
    north_mask = latitudes > np.median(latitudes)
    south_mask = ~north_mask
    
    north_avg_price = np.mean(y_test_original[north_mask]) * 100000
    south_avg_price = np.mean(y_test_original[south_mask]) * 100000
    
    print(f"   北部地区平均房价: ${north_avg_price:.0f}")
    print(f"   南部地区平均房价: ${south_avg_price:.0f}")
    
    # 按经度分析（东西方向）
    east_mask = longitudes > np.median(longitudes)
    west_mask = ~east_mask
    
    east_avg_price = np.mean(y_test_original[east_mask]) * 100000
    west_avg_price = np.mean(y_test_original[west_mask]) * 100000
    
    print(f"   东部地区平均房价: ${east_avg_price:.0f}")
    print(f"   西部地区平均房价: ${west_avg_price:.0f}")
    
    # 分析预测误差的地理分布
    errors = np.abs(y_test_original - predictions)
    
    print(f"\\n预测误差的地理模式:")
    print(f"   北部地区平均误差: ${np.mean(errors[north_mask])*100000:.0f}")
    print(f"   南部地区平均误差: ${np.mean(errors[south_mask])*100000:.0f}")
    print(f"   东部地区平均误差: ${np.mean(errors[east_mask])*100000:.0f}")
    print(f"   西部地区平均误差: ${np.mean(errors[west_mask])*100000:.0f}")


def demo_house_price_prediction(model, scaler_X, scaler_y, feature_names):
    """演示实际房价预测"""
    
    print("\\n假设您想评估一个特定区域的房价:")
    
    # 创建一个假设的房产区域
    sample_area = {
        'MedInc': 6.5,          # 收入中位数：6.5万美元
        'HouseAge': 15.0,       # 房屋年龄：15年
        'AveRooms': 6.2,        # 平均房间数：6.2个
        'AveBedrms': 1.1,       # 平均卧室数：1.1个  
        'Population': 3500,     # 人口：3500人
        'AveOccup': 3.2,        # 平均居住人数：3.2人
        'Latitude': 34.05,      # 纬度：34.05（洛杉矶附近）
        'Longitude': -118.25,   # 经度：-118.25
        'Rooms_per_household': 6.2/3.2,  # 每户房间数
        'Bedrooms_per_room': 1.1/6.2,    # 卧室比例
        'Population_density': 3500/3.2    # 人口密度代理
    }
    
    print("\\n🏘️ 目标区域特征:")
    interpretations = {
        'MedInc': '中上等收入社区',
        'HouseAge': '相对较新的房屋',
        'AveRooms': '宽敞的居住空间',
        'Latitude': '洛杉矶地区',
        'Longitude': '西海岸位置'
    }
    
    for feature, value in sample_area.items():
        if feature in interpretations:
            print(f"   {feature}: {value} ({interpretations[feature]})")
        else:
            print(f"   {feature}: {value:.2f}")
    
    # 准备预测数据
    sample_data = np.array([[sample_area[name] for name in feature_names]])
    sample_scaled = scaler_X.transform(sample_data)
    
    # 进行预测
    pred_scaled, uncertainty_scaled = model.predict(sample_scaled, return_uncertainty=True)
    
    # 反标准化
    predicted_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0] * 100000
    uncertainty = uncertainty_scaled[0] * scaler_y.scale_[0] * 100000
    
    print(f"\\n🔮 CausalQwen 的房价预测:")
    print(f"   预测房价: ${predicted_price:.0f}")
    print(f"   不确定性: ±${uncertainty:.0f}")
    print(f"   价格区间: ${predicted_price-uncertainty:.0f} - ${predicted_price+uncertainty:.0f}")
    
    # 给出投资建议
    if predicted_price > 400000:
        category = "高端"
    elif predicted_price > 250000:
        category = "中端"
    else:
        category = "入门级"
    
    confidence_level = "高" if uncertainty < 50000 else "中等" if uncertainty < 100000 else "低"
    
    print(f"\\n💡 投资分析:")
    print(f"   房价水平: {category}住宅区")
    print(f"   预测置信度: {confidence_level}")
    
    if uncertainty < 50000:
        print("   建议: 预测置信度高，适合投资决策参考")
    elif uncertainty < 100000:
        print("   建议: 预测有一定不确定性，建议结合其他信息")
    else:
        print("   建议: 不确定性较高，建议获取更多数据或专业评估")


def plot_detailed_analysis(y_true, y_pred, X_test, feature_names):
    """绘制详细分析图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 预测 vs 实际价格
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.5)
    
    # 理想预测线
    min_price = min(y_true.min(), y_pred.min())
    max_price = max(y_true.max(), y_pred.max())
    ax1.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8)
    
    ax1.set_xlabel('真实房价 (十万美元)')
    ax1.set_ylabel('预测房价 (十万美元)')
    ax1.set_title('预测 vs 实际房价')
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差分析
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('预测房价 (十万美元)')
    ax2.set_ylabel('残差')
    ax2.set_title('残差分析')
    ax2.grid(True, alpha=0.3)
    
    # 3. 房价分布
    ax3 = axes[1, 0]
    ax3.hist(y_true, bins=30, alpha=0.7, label='真实房价', density=True)
    ax3.hist(y_pred, bins=30, alpha=0.7, label='预测房价', density=True)
    ax3.set_xlabel('房价 (十万美元)')
    ax3.set_ylabel('密度')
    ax3.set_title('房价分布对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 地理分布（经纬度）
    ax4 = axes[1, 1]
    lat_idx = feature_names.index('Latitude')
    lon_idx = feature_names.index('Longitude')
    
    scatter = ax4.scatter(X_test[:, lon_idx], X_test[:, lat_idx], 
                         c=y_true, cmap='viridis', alpha=0.6)
    ax4.set_xlabel('经度')
    ax4.set_ylabel('纬度')
    ax4.set_title('房价地理分布')
    plt.colorbar(scatter, ax=ax4, label='房价 (十万美元)')
    
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 图表说明:")
    print("   - 左上: 预测准确性，点越接近红线越准确")
    print("   - 右上: 残差分析，点应随机分布在零线附近")
    print("   - 左下: 房价分布，两个分布应该相似")
    print("   - 右下: 地理分布，颜色表示房价高低")


if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    main()