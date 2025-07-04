import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.datasets import (
    fetch_california_housing, 
    load_diabetes, 
    load_linnerud,
    fetch_openml
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

def inject_shuffle_noise(
    y: np.ndarray,
    noise_ratio: float,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对一部分样本的标签进行全局随机洗牌，以切断其与特征的因果关系。
    这是项目MVP阶段唯一的因果标签异常注入逻辑。

    该方法基于一个理论上更优的策略：
    1. 创建一个全局洗牌后的标签副本 y_shuffled。
    2. 随机选择一部分索引。
    3. 将这些索引位置的原始标签替换为 y_shuffled 中对应位置的标签。
    这确保了被污染的标签与原始特征完全无关。

    Args:
        y: 原始标签数组。
        noise_ratio: 需要注入噪声的样本比例 (0.0 到 1.0)。
        random_state: 随机种子，用于复现。

    Returns:
        A tuple containing:
        - y_noisy (np.ndarray): 注入了洗牌噪声的标签数组。
        - noise_indices (np.ndarray): 被注入噪声的样本的原始索引。
    """
    if random_state is not None:
        np.random.seed(random_state)

    if not (0.0 <= noise_ratio <= 1.0):
        raise ValueError("noise_ratio 必须在 0.0 和 1.0 之间")

    n_samples = len(y)
    if noise_ratio == 0 or n_samples == 0:
        return y.copy(), np.array([], dtype=int)

    # 1. 创建一个全局洗牌后的标签副本 y'
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)

    # 2. 随机选择要污染的索引
    n_noisy = int(n_samples * noise_ratio)
    noise_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    # 3. 创建一个新的 y_noisy 向量
    y_noisy = y.copy()
    
    # 4. 在选定的索引处，用 y' 的值替换 y 的值
    y_noisy[noise_indices] = y_shuffled[noise_indices]
    
    return y_noisy, noise_indices

# =============================================================================
# 扩展回归数据集加载功能
# =============================================================================

# 扩展回归数据集配置
EXTENDED_REGRESSION_DATASETS = {
    # sklearn内置数据集
    'california_housing': {
        'source': 'sklearn',
        'loader': fetch_california_housing,
        'name': 'California Housing',
        'description': 'California housing prices (20,640 samples, 8 features)',
        'n_samples': 20640,
        'n_features': 8,
        'target_type': 'continuous'
    },
    'diabetes': {
        'source': 'sklearn',
        'loader': load_diabetes,
        'name': 'Diabetes',
        'description': 'Diabetes progression (442 samples, 10 features)',
        'n_samples': 442,
        'n_features': 10,
        'target_type': 'continuous'
    },
    'linnerud': {
        'source': 'sklearn',
        'loader': load_linnerud,
        'name': 'Linnerud Physical Exercise',
        'description': 'Physical exercise physiological data (20 samples, 3 features)',
        'n_samples': 20,
        'n_features': 3,
        'target_type': 'continuous',
        'multi_output': True
    },
    
    # OpenML数据集
    'boston_housing': {
        'source': 'openml',
        'name': 'Boston Housing (OpenML)',
        'description': 'Boston housing prices from OpenML (506 samples, 13 features)', 
        'openml_id': 531,  # Boston housing dataset ID
        'target_column': 'medv',
        'n_samples': 506,
        'n_features': 13,
        'target_type': 'continuous'
    },
    'auto_mpg': {
        'source': 'openml',
        'name': 'Auto MPG',
        'description': 'Auto fuel efficiency prediction (392 samples, 7 features)',
        'openml_id': 196,
        'target_column': 'mpg',
        'n_samples': 392,
        'n_features': 7,
        'target_type': 'continuous'
    },
    'wine_quality_red': {
        'source': 'openml',
        'name': 'Wine Quality (Red)',
        'description': 'Red wine quality rating (1599 samples, 11 features)',
        'openml_id': 287,
        'target_column': 'quality',
        'n_samples': 1599,
        'n_features': 11,
        'target_type': 'continuous'
    },
    'wine_quality_white': {
        'source': 'openml',
        'name': 'Wine Quality (White)',
        'description': 'White wine quality rating (4898 samples, 11 features)',
        'openml_id': 1497,
        'target_column': 'quality',
        'n_samples': 4898,
        'n_features': 11,
        'target_type': 'continuous'
    },
    'concrete_strength': {
        'source': 'openml',
        'name': 'Concrete Compressive Strength',
        'description': 'Concrete strength prediction (1030 samples, 8 features)',
        'openml_id': 4353,
        'target_column': 'class',
        'n_samples': 1030,
        'n_features': 8,
        'target_type': 'continuous'
    },
    'energy_efficiency_heating': {
        'source': 'openml',
        'name': 'Energy Efficiency (Heating)',
        'description': 'Building energy efficiency heating load (768 samples, 8 features)',
        'openml_id': 4514,
        'target_column': 'Y1',
        'n_samples': 768,
        'n_features': 8,
        'target_type': 'continuous'
    },
    'energy_efficiency_cooling': {
        'source': 'openml',
        'name': 'Energy Efficiency (Cooling)',
        'description': 'Building energy efficiency cooling load (768 samples, 8 features)',
        'openml_id': 4514,
        'target_column': 'Y2',
        'n_samples': 768,
        'n_features': 8,
        'target_type': 'continuous'
    },
    'abalone': {
        'source': 'openml',
        'name': 'Abalone Age',
        'description': 'Abalone age prediction (4177 samples, 8 features)',
        'openml_id': 183,
        'target_column': 'Rings',
        'n_samples': 4177,
        'n_features': 8,
        'target_type': 'continuous'
    }
}

def load_extended_regression_dataset(
    dataset_name: str, 
    random_state: int = 42,
    return_info: bool = True,
    handle_missing: str = 'auto',
    standardize_features: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """
    加载扩展的真实回归数据集
    
    支持sklearn内置数据集和OpenML数据集，提供统一的接口和数据预处理。
    
    Args:
        dataset_name: 数据集名称，必须在EXTENDED_REGRESSION_DATASETS中定义
        random_state: 随机种子，用于数据加载中的随机操作
        return_info: 是否返回数据集信息字典
        handle_missing: 缺失值处理方式 ('auto', 'drop', 'fill_mean', 'fill_median')
        standardize_features: 是否标准化特征
        
    Returns:
        - X (np.ndarray): 特征矩阵
        - y (np.ndarray): 目标变量
        - dataset_info (Dict, optional): 数据集信息字典
        
    Raises:
        ValueError: 如果数据集名称不支持
        RuntimeError: 如果数据加载失败
    """
    if dataset_name not in EXTENDED_REGRESSION_DATASETS:
        available_datasets = list(EXTENDED_REGRESSION_DATASETS.keys())
        raise ValueError(f"不支持的数据集: {dataset_name}. 可选择: {available_datasets}")
    
    dataset_config = EXTENDED_REGRESSION_DATASETS[dataset_name]
    
    print(f"📊 加载扩展回归数据集: {dataset_config['name']}")
    print(f"📋 数据集描述: {dataset_config['description']}")
    
    try:
        if dataset_config['source'] == 'sklearn':
            X, y = _load_sklearn_dataset(dataset_config, random_state)
        elif dataset_config['source'] == 'openml':
            X, y = _load_openml_dataset(dataset_config, random_state)
        else:
            raise ValueError(f"不支持的数据源: {dataset_config['source']}")
        
        print(f"✅ 数据集加载成功: {X.shape[0]} samples, {X.shape[1]} features")
        
        # 数据预处理
        X, y = _preprocess_dataset(X, y, handle_missing, standardize_features)
        
        if return_info:
            dataset_info = _create_dataset_info(dataset_config, X, y)
            return X, y, dataset_info
        else:
            return X, y
            
    except Exception as e:
        raise RuntimeError(f"数据集 {dataset_name} 加载失败: {str(e)}")

def _load_sklearn_dataset(dataset_config: Dict[str, Any], random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """加载sklearn内置数据集"""
    loader = dataset_config['loader']
    
    if 'linnerud' in dataset_config.get('name', '').lower():
        # Linnerud是多输出，取第一个输出
        data = loader()
        X, y = data.data, data.target
        if y.ndim > 1:
            y = y[:, 0]  # 取第一个目标变量
    else:
        data = loader()
        X, y = data.data, data.target
    
    return X, y

def _load_openml_dataset(dataset_config: Dict[str, Any], random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """加载OpenML数据集"""
    openml_id = dataset_config['openml_id']
    target_column = dataset_config.get('target_column')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')
    
    # 处理特征和目标
    X = data.data
    
    # 处理目标变量
    if target_column and target_column in data.data.columns:
        # 目标变量在特征矩阵中
        y = data.data[target_column]
        X = data.data.drop(columns=[target_column])
    else:
        # 目标变量单独存在
        y = data.target
    
    # 转换为numpy数组
    X = _convert_to_numeric_array(X, 'features')
    y = _convert_to_numeric_array(y, 'target')
    
    return X, y

def _convert_to_numeric_array(data, data_type: str) -> np.ndarray:
    """将pandas DataFrame/Series转换为数值型numpy数组"""
    if isinstance(data, pd.DataFrame):
        # 创建副本避免警告
        data = data.copy()
        
        # 处理分类变量
        for col in data.columns:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                # 使用LabelEncoder处理分类变量
                le = LabelEncoder()
                data.loc[:, col] = le.fit_transform(data[col].astype(str))
        
        return data.values.astype(np.float64)
    
    elif isinstance(data, pd.Series):
        # 处理目标变量
        if data.dtype == 'object' or data.dtype.name == 'category':
            le = LabelEncoder() 
            data = le.fit_transform(data.astype(str))
        
        return np.array(data, dtype=np.float64)
    
    else:
        return np.array(data, dtype=np.float64)

def _preprocess_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    handle_missing: str,
    standardize_features: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """数据预处理"""
    
    # 处理缺失值
    if handle_missing != 'auto':
        X, y = _handle_missing_values(X, y, handle_missing)
    else:
        # 自动处理：如果有缺失值则使用均值填充
        if np.isnan(X).any() or np.isnan(y).any():
            X, y = _handle_missing_values(X, y, 'fill_mean')
    
    # 特征标准化
    if standardize_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y

def _handle_missing_values(
    X: np.ndarray, 
    y: np.ndarray, 
    method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """处理缺失值"""
    
    if method == 'drop':
        # 删除包含缺失值的行
        missing_mask = np.isnan(X).any(axis=1) | np.isnan(y)
        valid_indices = ~missing_mask
        return X[valid_indices], y[valid_indices]
    
    elif method == 'fill_mean':
        # 用均值填充
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]
        
        if np.isnan(y).any():
            y_mean = np.nanmean(y)
            y[np.isnan(y)] = y_mean
            
    elif method == 'fill_median':
        # 用中位数填充
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_medians[i]
        
        if np.isnan(y).any():
            y_median = np.nanmedian(y)
            y[np.isnan(y)] = y_median
    
    return X, y

def _create_dataset_info(
    dataset_config: Dict[str, Any], 
    X: np.ndarray, 
    y: np.ndarray
) -> Dict[str, Any]:
    """创建数据集信息字典"""
    return {
        'name': dataset_config['name'],
        'description': dataset_config['description'],
        'source': dataset_config['source'],
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'target_type': dataset_config['target_type'],
        'y_min': float(np.min(y)),
        'y_max': float(np.max(y)),
        'y_mean': float(np.mean(y)),
        'y_std': float(np.std(y)),
        'has_missing': np.isnan(X).any() or np.isnan(y).any()
    }

def list_available_regression_datasets() -> None:
    """列出所有可用的回归数据集"""
    print("🎯 可用的扩展回归数据集:")
    print("=" * 70)
    
    sklearn_datasets = []
    openml_datasets = []
    
    for name, config in EXTENDED_REGRESSION_DATASETS.items():
        if config['source'] == 'sklearn':
            sklearn_datasets.append((name, config))
        else:
            openml_datasets.append((name, config))
    
    if sklearn_datasets:
        print("\n📦 sklearn内置数据集:")
        for name, config in sklearn_datasets:
            print(f"  • {name}: {config['description']}")
    
    if openml_datasets:
        print("\n🌐 OpenML数据集:")
        for name, config in openml_datasets:
            print(f"  • {name}: {config['description']}")
    
    print(f"\n✅ 总计 {len(EXTENDED_REGRESSION_DATASETS)} 个数据集可用")

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """获取指定数据集的详细信息"""
    if dataset_name not in EXTENDED_REGRESSION_DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    return EXTENDED_REGRESSION_DATASETS[dataset_name].copy()