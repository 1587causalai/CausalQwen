"""
基准测试方法配置模块

定义所有传统机器学习方法的默认参数配置，支持通过配置灵活选择和调整基准方法。
"""

# 🧠 统一神经网络参数配置 - 在此处修改所有神经网络方法的共同参数
# =========================================================================
# 🔧 修改这些变量会影响所有神经网络方法的参数！
NN_HIDDEN_SIZES = (128, 64, 32)                 # 神经网络隐藏层结构
NN_MAX_EPOCHS = 3000                            # 最大训练轮数
NN_LEARNING_RATE = 0.01                         # 学习率
NN_PATIENCE = 200                               # 早停patience
NN_TOLERANCE = 1e-4                             # 早停tolerance
# =========================================================================

# 默认基准方法配置
DEFAULT_METHOD_CONFIGS = {
    # === 神经网络方法 ===
    'sklearn_mlp': {
        'name': 'sklearn MLP',
        'type': 'neural_network',
        'params': {
            'hidden_layer_sizes': NN_HIDDEN_SIZES,
            'max_iter': NN_MAX_EPOCHS,
            'learning_rate_init': NN_LEARNING_RATE,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': NN_PATIENCE,
            'alpha': 0.0,
            'random_state': 42,
            'tol': NN_TOLERANCE,
        }
    },
    
    'pytorch_mlp': {
        'name': 'PyTorch MLP',
        'type': 'neural_network',
        'params': {
            'hidden_layer_sizes': NN_HIDDEN_SIZES,
            'max_iter': NN_MAX_EPOCHS,
            'learning_rate': NN_LEARNING_RATE,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': NN_PATIENCE,
            'tol': NN_TOLERANCE,
            'alpha': 0.0,
            'random_state': 42,
            'batch_size': None
        }
    },
    
    'mlp_huber': {
        'name': 'MLP Huber',
        'type': 'robust_neural_network',
        'params': {
            'hidden_layer_sizes': NN_HIDDEN_SIZES,
            'max_iter': NN_MAX_EPOCHS,
            'learning_rate': NN_LEARNING_RATE,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': NN_PATIENCE,
            'tol': NN_TOLERANCE,
            'alpha': 0.0,
            'random_state': 42,
            'batch_size': None,
            'delta': 1.0  # Huber损失的delta参数
        }
    },
    
    'mlp_pinball_median': {
        'name': 'MLP Pinball Median',
        'type': 'robust_neural_network', 
        'params': {
            'hidden_layer_sizes': NN_HIDDEN_SIZES,
            'max_iter': NN_MAX_EPOCHS,
            'learning_rate': NN_LEARNING_RATE,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': NN_PATIENCE,
            'tol': NN_TOLERANCE,
            'alpha': 0.0,
            'random_state': 42,
            'batch_size': None,
            'quantile': 0.5  # 中位数回归 (50%分位数)
        }
    },
    
    'mlp_cauchy': {
        'name': 'MLP Cauchy',
        'type': 'robust_neural_network',
        'params': {
            'hidden_layer_sizes': NN_HIDDEN_SIZES,
            'max_iter': NN_MAX_EPOCHS,
            'learning_rate': NN_LEARNING_RATE,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': NN_PATIENCE,
            'tol': NN_TOLERANCE,
            'alpha': 0.0,
            'random_state': 42,
            'batch_size': None
        }
    },
    
    # === 集成方法 ===
    'random_forest': {
        'name': 'Random Forest',
        'type': 'ensemble',
        'params': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    
    'extra_trees': {
        'name': 'Extra Trees',
        'type': 'ensemble',
        'params': {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    },
    
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'type': 'ensemble',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': 42
        }
    },
    
    'xgboost': {
        'name': 'XGBoost',
        'type': 'boosting',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    },
    
    'lightgbm': {
        'name': 'LightGBM',
        'type': 'boosting',
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
    },
    
    'catboost': {
        'name': 'CatBoost',
        'type': 'boosting',
        'params': {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'verbose': False,
            'thread_count': -1
        }
    },
    
    # === 支持向量机 ===
    'svm_rbf': {
        'name': 'SVM (RBF)',
        'type': 'svm',
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'tol': 1e-3,
            'max_iter': 1000
        }
    },
    
    'svm_linear': {
        'name': 'SVM (Linear)',
        'type': 'svm',
        'params': {
            'kernel': 'linear',
            'C': 1.0,
            'tol': 1e-3,
            'max_iter': 1000
        }
    },
    
    'svm_poly': {
        'name': 'SVM (Poly)',
        'type': 'svm',
        'params': {
            'kernel': 'poly',
            'degree': 3,
            'C': 1.0,
            'gamma': 'scale',
            'tol': 1e-3,
            'max_iter': 1000
        }
    },
    
    # === 线性方法 ===
    'linear_regression': {
        'name': 'Linear Regression',
        'type': 'linear',
        'params': {}
    },
    
    'ridge': {
        'name': 'Ridge Regression',
        'type': 'linear',
        'params': {
            'alpha': 1.0,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': 42
        }
    },
    
    'lasso': {
        'name': 'Lasso Regression',
        'type': 'linear',
        'params': {
            'alpha': 1.0,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': 42
        }
    },
    
    'elastic_net': {
        'name': 'Elastic Net',
        'type': 'linear',
        'params': {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': 42
        }
    },
    
    # === K近邻 ===
    'knn': {
        'name': 'K-Nearest Neighbors',
        'type': 'neighbor',
        'params': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'n_jobs': -1
        }
    },
    
    # === 决策树 ===
    'decision_tree': {
        'name': 'Decision Tree',
        'type': 'tree',
        'params': {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }
}

# 预定义方法组合
METHOD_GROUPS = {
    # 基础组合（快速测试）
    'basic': ['sklearn_mlp', 'pytorch_mlp', 'random_forest', 'xgboost'],
    
    # 神经网络组合
    'neural': ['sklearn_mlp', 'pytorch_mlp', 'mlp_huber', 'mlp_pinball_median', 'mlp_cauchy'],
    
    # 集成方法组合
    'ensemble': ['random_forest', 'extra_trees', 'gradient_boosting'],
    
    # 提升方法组合
    'boosting': ['xgboost', 'lightgbm', 'catboost'],
    
    # 支持向量机组合
    'svm': ['svm_rbf', 'svm_linear', 'svm_poly'],
    
    # 线性方法组合
    'linear': ['linear_regression', 'ridge', 'lasso', 'elastic_net'],
    
    # 全面测试组合
    'comprehensive': [
        'sklearn_mlp', 'pytorch_mlp', 'random_forest', 'extra_trees', 
        'gradient_boosting', 'xgboost', 'lightgbm', 'svm_rbf', 'knn'
    ],
    
    # 竞争性基准组合（最强方法）
    'competitive': ['pytorch_mlp', 'random_forest', 'xgboost', 'lightgbm', 'svm_rbf'],
    
    # 轻量级测试组合（快速验证）
    'lightweight': ['sklearn_mlp', 'random_forest', 'xgboost']
}

# 任务类型特定的推荐方法
TASK_SPECIFIC_RECOMMENDATIONS = {
    'regression': {
        'default': ['sklearn_mlp', 'pytorch_mlp', 'random_forest', 'xgboost', 'svm_rbf'],
        'robust': ['mlp_huber', 'mlp_pinball_median', 'mlp_cauchy', 'random_forest', 'extra_trees', 'gradient_boosting'],  # 对异常值鲁棒
        'fast': ['sklearn_mlp', 'random_forest', 'ridge'],  # 快速训练
        'accurate': ['pytorch_mlp', 'xgboost', 'lightgbm', 'svm_rbf']  # 高精度
    },
    'classification': {
        'default': ['sklearn_mlp', 'pytorch_mlp', 'random_forest', 'xgboost', 'svm_rbf'],
        'robust': ['random_forest', 'extra_trees', 'gradient_boosting', 'svm_rbf'],
        'fast': ['sklearn_mlp', 'random_forest', 'knn'],
        'accurate': ['pytorch_mlp', 'xgboost', 'lightgbm', 'svm_rbf']
    }
}

def get_method_config(method_name):
    """获取指定方法的配置"""
    return DEFAULT_METHOD_CONFIGS.get(method_name, None)

def get_method_group(group_name):
    """获取预定义的方法组合"""
    return METHOD_GROUPS.get(group_name, [])

def get_task_recommendations(task_type, recommendation_type='default'):
    """获取任务特定的推荐方法"""
    task_recs = TASK_SPECIFIC_RECOMMENDATIONS.get(task_type, {})
    return task_recs.get(recommendation_type, task_recs.get('default', []))

def list_available_methods():
    """列出所有可用的基准方法"""
    return list(DEFAULT_METHOD_CONFIGS.keys())

def list_available_groups():
    """列出所有可用的方法组合"""
    return list(METHOD_GROUPS.keys())

def validate_methods(method_list):
    """验证方法列表是否都是可用的"""
    available = set(DEFAULT_METHOD_CONFIGS.keys())
    invalid = [m for m in method_list if m not in available]
    
    if invalid:
        raise ValueError(f"未知的基准方法: {invalid}. 可用方法: {list(available)}")
    
    return True

def expand_method_groups(method_list):
    """展开方法组合为具体方法列表"""
    expanded = []
    
    for item in method_list:
        if item.startswith('group:'):
            # 处理组合引用，如 'group:basic'
            group_name = item[6:]  # 去掉 'group:' 前缀
            if group_name in METHOD_GROUPS:
                expanded.extend(METHOD_GROUPS[group_name])
            else:
                raise ValueError(f"未知的方法组合: {group_name}")
        else:
            # 普通方法名
            expanded.append(item)
    
    # 去重并保持顺序
    seen = set()
    result = []
    for method in expanded:
        if method not in seen:
            seen.add(method)
            result.append(method)
    
    return result